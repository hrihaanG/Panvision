"""Doctor workspace: upload a CT before showing its report."""
import hashlib
import html
import tempfile
import csv
import io
from pathlib import Path

import numpy as np
import streamlit as st

from app.api_client import request, sign_out, APIError, SessionExpired
from app.auth import require_doctor, idle_watchdog
from app.data import (gradcam_panels, upload_display_mode, upload_gradcam_slice, upload_lesion_slice,
                      upload_pancreas_slice, upload_rgb_slice)
from app.validation import read_ct


def navigate(page):
    # Safe from both callbacks and an upload-validation handler after rendering.
    st.session_state['_pending_workspace_page'] = page


def clear_upload():
    for key in ('upload_review', 'uploaded_report_pdf', 'uploaded_report_notes',
                'saved_report_notes', 'uploaded_patient_number', 'image_display_mode'):
        st.session_state.pop(key, None)


def accept_upload(name, content):
    """Validate before replacing the current report; never attach dataset labels."""
    clear_upload()
    suffix = '.nii.gz' if name.lower().endswith('.nii.gz') else Path(name).suffix.lower()
    if suffix not in ('.nii', '.nii.gz', '.mha'):
        raise ValueError('Choose a .nii, .nii.gz, or self-contained .mha scan.')
    with tempfile.TemporaryDirectory(prefix='panvision-upload-') as folder:
        path = Path(folder) / ('scan' + suffix)
        path.write_bytes(content)
        array, spacing = read_ct(path)
    st.session_state['upload_review'] = {
        'digest': hashlib.sha256(content).hexdigest(), 'name': name,
        'array': array, 'spacing': spacing, 'content': content,
    }


def run_full_analysis(reviewed):
    response = request('POST', f'/uploads/{reviewed["upload_id"]}/analysis',
                       files={'file': (reviewed['name'], reviewed['content'])}, timeout=1800)
    try:
        with np.load(io.BytesIO(response.content), allow_pickle=False) as result:
            pancreas = result['pancreas'].astype(bool)
            if pancreas.shape != reviewed['array'].shape:
                raise ValueError('Mask shape does not match the scan.')
            lesion = result['lesion'].astype(bool)
            if lesion.shape != reviewed['array'].shape:
                raise ValueError('Lesion mask shape does not match the scan.')
            gradcam = np.asarray(result['gradcam'], dtype=np.float32)
            gradcam_slice = int(result['gradcam_slice'])
            probability = float(result['pdac_probability'])
            threshold = float(result['threshold'])
            if (gradcam.shape != reviewed['array'].shape[1:] or not np.isfinite(gradcam).all() or
                    not 0 <= gradcam_slice < len(reviewed['array'])):
                raise ValueError('Grad-CAM does not align with the scan.')
            if not 0 <= probability <= 1 or not 0 < threshold < 1:
                raise ValueError('Classification scores are invalid.')
            neighbors = [
                {'patient_id': str(patient), 'label': int(label), 'similarity': float(similarity)}
                for patient, label, similarity in zip(
                    result['neighbor_ids'], result['neighbor_labels'],
                    result['neighbor_similarities'])
            ]
            if len(neighbors) > 3 or any(row['label'] not in (0, 1) or
                                         not -1 <= row['similarity'] <= 1 for row in neighbors):
                raise ValueError('Similar-case results are invalid.')
            attention = {}
            if 'gradcam_crop' in result.files:  # newer backends also send the crop view
                crop = np.asarray(result['gradcam_crop'], dtype=np.uint8)
                crop_heat = np.asarray(result['gradcam_crop_heat'], dtype=np.float32)
                slice_probability = float(result['gradcam_probability'])
                if (crop.shape != (224, 224) or crop_heat.shape != (224, 224) or
                        not np.isfinite(crop_heat).all() or not 0 <= slice_probability <= 1):
                    raise ValueError('Grad-CAM crop does not match the classifier input.')
                attention = {'gradcam_crop': crop, 'gradcam_crop_heat': crop_heat,
                             'gradcam_probability': slice_probability}
            reviewed.update({
                **attention,
                'pancreas_mask': pancreas,
                'lesion_mask': lesion,
                'gradcam': gradcam,
                'gradcam_slice': gradcam_slice,
                'predicted_class': str(result['predicted_class']),
                'pdac_probability': probability,
                'classification_threshold': threshold,
                'similar_cases': neighbors,
            })
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError('The backend returned invalid inference results.') from exc
    reviewed.pop('analysis_error', None)


def heading(title, subtitle):
    st.markdown(f'<h1>{html.escape(title)}</h1><p class="subtitle">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def metric_card(label, value, detail=''):
    st.markdown(f'<div class="metric-card"><div class="eyebrow">{html.escape(label)}</div>'
                f'<div class="metric-value">{html.escape(str(value))}</div>'
                f'<div class="metric-detail">{html.escape(detail)}</div></div>', unsafe_allow_html=True)


def placeholder(title, detail, symbol='◌'):
    st.markdown(f'<div class="scan-placeholder"><span>{symbol}</span><strong>{html.escape(title)}</strong>'
                f'<p>{html.escape(detail)}</p></div>', unsafe_allow_html=True)


def scan_strip(reviewed):
    array = reviewed['array'] if reviewed else None
    name = reviewed['name'] if reviewed else 'No scan selected'
    items = [('Scan', name), ('Slices', str(len(array)) if reviewed else '—'),
             ('File validation', 'Passed' if reviewed else 'Awaiting upload'),
             ('Model analysis', 'Complete' if reviewed and reviewed.get('predicted_class')
              else 'Not run' if reviewed else 'Awaiting upload')]
    st.markdown('<div class="scan-strip">'+''.join(
        f'<div><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in items)+'</div>', unsafe_allow_html=True)


def imaging_panels(reviewed, key):
    if not reviewed:
        with st.container(border=True):
            placeholder('Upload a CT scan', 'The CT, pancreas overlay, and model attention will appear here.', '⊕')
        return
    array = reviewed['array']
    display_mode = st.selectbox('Image display', ['auto', 'normalized', 'hu'],
                                format_func=lambda value: {'auto':'Automatic', 'normalized':'Normalized (0–1)',
                                                           'hu':'CT window (−100 to 300 HU)'}[value],
                                key='image_display_mode')
    resolved = upload_display_mode(array) if display_mode == 'auto' else display_mode
    reviewed['display_mode'] = display_mode
    slice_key = f'{key}_slice_{reviewed["digest"]}'
    if slice_key not in st.session_state:
        mask = reviewed.get('pancreas_mask')
        st.session_state[slice_key] = (int(mask.sum(axis=(1, 2)).argmax())
                                       if mask is not None and mask.any() else len(array) // 2)
    z = st.slider('CT slice', 0, len(array) - 1, key=slice_key)
    ct_column, pancreas_column, lesion_column = st.columns(3, gap='medium')
    with ct_column, st.container(border=True):
        st.markdown('##### Input CT scan')
        st.image(upload_rgb_slice(array, z, resolved), width='stretch')
        st.caption(f'Slice {z + 1} / {len(array)} · Uploaded CT')
    with pancreas_column, st.container(border=True):
        st.markdown('##### Predicted pancreas')
        if reviewed.get('pancreas_mask') is not None:
            st.image(upload_pancreas_slice(array, reviewed['pancreas_mask'], z, resolved), width='stretch')
            st.caption(f'Green: model-estimated pancreas · Slice {z + 1} / {len(array)}')
        else:
            placeholder('Analysis unavailable', 'The pancreas model did not return an overlay.')
    with lesion_column, st.container(border=True):
        st.markdown('##### Predicted lesion')
        if reviewed.get('lesion_mask') is not None:
            st.image(upload_lesion_slice(array, reviewed['lesion_mask'], z, resolved), width='stretch')
            st.caption(f'Coral: model-estimated lesion · Slice {z + 1} / {len(array)}')
        else:
            placeholder('Analysis unavailable', 'Run the analysis to create the lesion overlay.')
    with st.container(border=True):
        st.markdown('##### Model attention · Grad-CAM')
        if reviewed.get('gradcam') is not None:
            attention_slice = int(reviewed['gradcam_slice'])
            if reviewed.get('gradcam_crop') is not None:
                # Crop | heat map | overlay, the same three pictures as in the experiment notebooks.
                pictures = gradcam_panels(reviewed['gradcam_crop'], reviewed['gradcam_crop_heat'])
                for column, title, picture in zip(st.columns(3, gap='small'),
                                                  ['Pancreas crop', 'Grad-CAM heat map', 'Overlay'], pictures):
                    with column:
                        st.markdown(f'**{title}**')
                        st.image(picture, width='stretch')
                st.caption(f'Most suspicious slice ({attention_slice + 1} / {len(array)}): chance of PDAC '
                           f'{reviewed["gradcam_probability"]:.2f}. 144 mm window around the predicted pancreas. '
                           'Blue = low model attention, red = high. Coarse by design; not a lesion segmentation.')
                with st.expander('Show on the full CT slice'):
                    st.image(upload_gradcam_slice(array, reviewed['gradcam'], attention_slice, resolved),
                             width='stretch')
            else:
                st.image(upload_gradcam_slice(array, reviewed['gradcam'], attention_slice, resolved),
                         width='stretch')
                st.caption(f'Slice {attention_slice + 1} / {len(array)}. Grad-CAM is computed on the DenseNet pancreas crop '
                           'and mapped back to the uploaded CT for context. Warmer color means greater model attention.')
        else:
            placeholder('Analysis unavailable', 'Grad-CAM was not returned by the classifier.')
    st.caption('All overlays are research model outputs and are not a diagnosis.')


def result_cards(reviewed):
    prediction = reviewed.get('predicted_class') if reviewed else None
    probability = reviewed.get('pdac_probability') if reviewed else None
    threshold = reviewed.get('classification_threshold') if reviewed else None
    similar = reviewed.get('similar_cases', []) if reviewed else []
    for column, title, value, detail in zip(st.columns(3),
            ['PDAC classification', 'PDAC probability', 'Decision threshold'],
            [prediction or ('Not analyzed' if reviewed else 'No result yet'),
             f'{probability:.1%}' if probability is not None else 'Not available',
             f'{threshold:.1%}' if threshold is not None else 'Not available'],
            [f'PDAC probability {probability:.1%}; research model output only.' if probability is not None
             else 'DenseNet classification was not completed.',
             'Patient-level mean probability across predicted-pancreas crops.',
             'Frozen validation threshold used to convert probability into a class.']):
        with column, st.container(border=True):
            st.markdown(f'##### {title}')
            st.markdown(f'<div class="result-status">{value}</div>', unsafe_allow_html=True)
            st.caption(detail)
    if similar:
        st.markdown('##### Similar training cases')
        similar_table(similar)


def similar_table(similar):
    st.dataframe([
        {'Rank': rank, 'Patient ID': row['patient_id'],
         'Dataset label': 'PDAC' if row['label'] == 1 else 'Non-PDAC',
         'Cosine similarity': f'{row["similarity"]:.4f}'}
        for rank, row in enumerate(similar, 1)
    ], hide_index=True, width='stretch')


def similar_cases():
    reviewed = st.session_state.get('upload_review')
    heading('Similar cases', 'Nearest training cases to your uploaded scan by DenseNet embedding.')
    if not reviewed:
        placeholder('No scan uploaded yet', 'Upload and analyze a CT scan to retrieve similar training cases.', '⊕')
        st.button('Upload CT scan', type='primary', on_click=navigate, args=('Upload CT',), key='similar_upload')
        return
    scan_strip(reviewed)
    if not reviewed.get('predicted_class'):
        placeholder('Analysis not run', 'Run the analysis from the case report to retrieve similar training cases.')
        st.button('Open case report', type='primary', on_click=navigate, args=('Case Report',), key='similar_report')
        return
    similar = reviewed.get('similar_cases', [])
    if not similar:
        placeholder('No similar cases returned', 'The reference bank had no training cases for this classifier version.')
        return
    pdac = sum(row['label'] == 1 for row in similar)
    left, middle, right = st.columns(3)
    with left:
        metric_card('This scan', reviewed['predicted_class'], f'PDAC probability {reviewed["pdac_probability"]:.1%}')
    with middle:
        metric_card('Nearest cases labelled PDAC', f'{pdac} of {len(similar)}', 'Dataset labels of the training references')
    with right:
        metric_card('Closest match', f'{similar[0]["similarity"]:.4f}', 'Cosine similarity of the top neighbour')
    with st.container(border=True):
        st.markdown('##### Nearest training cases')
        similar_table(similar)
        st.caption('Neighbours come from the training split only and are matched on the classifier embedding, '
                   'not on the image directly. Research use only; this is not a diagnosis.')


def home(doctor):
    reviewed = st.session_state.get('upload_review')
    title, action = st.columns([4, 1])
    with title:
        heading('Welcome to PanVision', 'Review CT scans, inspect model outputs, and prepare a structured case report.')
    with action:
        st.button('＋ Upload new CT', type='primary', on_click=navigate, args=('Upload CT',), width='stretch')
    scan_strip(reviewed)
    imaging_panels(reviewed, 'home')
    result_cards(reviewed)
    left, middle, right = st.columns([1.1, 1, 1])
    with left, st.container(border=True):
        st.markdown('##### Key findings')
        if reviewed:
            st.write('The uploaded scan passed structural validation.')
            if reviewed.get('predicted_class'):
                st.caption('Pancreas segmentation, classification, similar-case retrieval, and Grad-CAM are complete.')
            else:
                st.caption('Model analysis did not complete. File validation does not establish a diagnosis.')
        else:
            st.write('No scan has been uploaded.')
            st.caption('Start by uploading a CT. Findings and reports remain empty until a valid scan is available.')
    with middle, st.container(border=True):
        st.markdown('##### Scan information')
        st.write('**Source:** Uploaded scan' if reviewed else '**Source:** —')
        st.write('**Pancreas segmentation:** ' + ('Complete' if reviewed and reviewed.get('pancreas_mask') is not None
                                                  else 'Not run' if reviewed else 'Awaiting upload'))
        st.write('**Classification:** ' + (reviewed.get('predicted_class', 'Not run') if reviewed else 'Awaiting upload'))
        st.caption('Temporary review · Research use only')
    with right, st.container(border=True):
        st.markdown('##### Actions')
        st.button('Open case report', disabled=not reviewed, on_click=navigate, args=('Case Report',), width='stretch')
        st.button('New analysis', on_click=navigate, args=('Upload CT',), width='stretch')


def upload_ct():
    heading('New analysis', 'Choose a CT scan, validate it, then review and start model analysis.')
    st.markdown('<div class="steps"><b>01 · Upload CT</b><span>02 · Validate</span><span>03 · Run analysis</span></div>', unsafe_allow_html=True)
    with st.container(border=True):
        uploaded = st.file_uploader('Choose your CT scan', type=['nii', 'gz', 'mha'],
                                    on_change=clear_upload, key='ct_upload')
        st.caption('Uploader, filename, validation status and timestamps are saved to your history. CT files stay temporary.')
        if uploaded is not None:
            digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
            current = st.session_state.get('upload_review')
            if current and current.get('digest') == digest:
                st.info('This scan is already validated. Open Case Report to run or review analysis.')
                return
            try:
                with st.spinner('Validating the CT scan…'):
                    content = uploaded.getvalue()
                    validation = request('POST', '/uploads/validate', files={'file': (uploaded.name, content)}).json()
                    accept_upload(uploaded.name, content)
                    reviewed = st.session_state['upload_review']
                    reviewed['upload_id'] = validation.get('upload_id')
            except (APIError, ValueError, RuntimeError, OSError) as exc:
                if not st.session_state.get('upload_review'):
                    clear_upload()
                st.error(str(exc) if isinstance(exc, APIError) else
                         'Unable to read this scan. Use a valid finite 3D NIfTI or self-contained MHA file.')
            else:
                navigate('Case Report')
                st.rerun()


def uploaded_report():
    reviewed = st.session_state.get('upload_review')
    if not reviewed:
        heading('No scan uploaded yet', 'Upload a CT scan to create its report.')
        st.button('Upload CT scan', type='primary', on_click=navigate, args=('Upload CT',))
        return
    heading('Case report', 'Scan evidence, model status, and review notes.')
    scan_strip(reviewed)
    if reviewed.get('analysis_error'):
        st.error(f'Analysis could not complete: {reviewed["analysis_error"]}')
    elif reviewed.get('predicted_class'):
        st.success('Analysis complete: pancreas and lesion segmentation, DenseNet classification, KNN retrieval, and Grad-CAM.')
    else:
        st.info('The scan passed validation. Start analysis when you are ready.')
    if not reviewed.get('predicted_class'):
        if st.button('Run full AI analysis', type='primary', width='stretch'):
            try:
                with st.spinner('Running pancreas and lesion segmentation, DenseNet, KNN, and Grad-CAM…'):
                    run_full_analysis(reviewed)
            except (APIError, ValueError) as exc:
                if isinstance(exc, SessionExpired):
                    raise
                reviewed['analysis_error'] = str(exc)
                st.error(f'Analysis could not complete: {exc}')
            else:
                st.rerun()
    array, spacing = reviewed['array'], reviewed['spacing']
    imaging_panels(reviewed, 'report')
    result_cards(reviewed)
    left, right = st.columns([1, 1])
    with left, st.container(border=True):
        st.subheader('Scan geometry')
        st.write(f'**Slices:** {len(array)}')
        st.write(f'**Image size:** {array.shape[2]} × {array.shape[1]} pixels')
        st.write('**Voxel spacing:** ' + ' × '.join(f'{v:.2f}' for v in spacing) + ' mm')
        st.caption('Display window: −100 to 300 HU. Uploaded scan values require acquisition calibration.')
    with right, st.container(border=True):
        st.subheader('Review notes')
        patient_number = st.text_input(
            'Patient number (optional)', key='uploaded_patient_number', max_chars=100,
            placeholder='Enter manually if unavailable from the scan',
        )
        notes = st.text_area('Notes for this report', key='uploaded_report_notes', max_chars=10000, height=200,
                             placeholder='Add your observations…')
        notes_status = st.empty()
        if st.button('Save review notes', disabled=not reviewed.get('upload_id'), width='stretch'):
            request('PUT', f'/uploads/{reviewed["upload_id"]}/notes', json={'notes':notes})
            st.session_state['saved_report_notes'] = notes
        st.write('**Model classification:** ' + reviewed.get('predicted_class', 'Not available'))
        st.write('**Pancreas segmentation:** ' + ('Complete' if reviewed.get('pancreas_mask') is not None else 'Not run'))
        st.write('**Lesion segmentation:** ' + ('Complete' if reviewed.get('lesion_mask') is not None else 'Not run'))
        st.write('**Grad-CAM:** ' + ('Complete' if reviewed.get('gradcam') is not None else 'Not available'))
        pancreas_mask = reviewed.get('pancreas_mask')
        lesion_mask = reviewed.get('lesion_mask')
        gradcam = reviewed.get('gradcam')
        pdf_key = (reviewed['digest'], notes, reviewed.get('display_mode', 'auto'),
                   reviewed.get('predicted_class'), reviewed.get('pdac_probability'),
                   reviewed.get('classification_threshold'), id(pancreas_mask), id(lesion_mask), id(gradcam))
        if st.button('Prepare PDF', type='primary', width='stretch'):
            with st.spinner('Preparing report…'):
                report_data = {'notes':notes, 'display_mode':reviewed.get('display_mode','auto'),
                               'patient_number':patient_number.strip()}
                if reviewed.get('upload_id'):
                    report_data['upload_id'] = reviewed['upload_id']
                if reviewed.get('predicted_class'):
                    report_data['gradcam_slice'] = reviewed['gradcam_slice']
                files = {'file': (reviewed['name'], reviewed['content'])}
                if pancreas_mask is not None:
                    packed = np.packbits(pancreas_mask.reshape(-1)).tobytes()
                    files['pancreas_mask'] = ('pancreas.mask', packed)
                if gradcam is not None:
                    buffer = io.BytesIO()
                    np.save(buffer, np.asarray(gradcam, dtype=np.float32), allow_pickle=False)
                    files['gradcam'] = ('gradcam.npy', buffer.getvalue())
                pdf = request('POST', '/uploads/report.pdf',
                              files=files, data=report_data).content
                st.session_state['uploaded_report_pdf'] = (pdf_key, pdf)
                st.session_state['saved_report_notes'] = notes
        if notes != st.session_state.get('saved_report_notes', ''):
            notes_status.warning('Unsaved changes. Save notes or prepare the PDF to keep them with this scan.')
        elif notes:
            notes_status.caption('✓ Notes saved to this scan’s history.')
        else:
            notes_status.caption('No review notes added yet.')
        prepared = st.session_state.get('uploaded_report_pdf')
        if prepared and prepared[0] == pdf_key:
            st.download_button('Download report · PDF', prepared[1], 'panvision-ct-report.pdf', 'application/pdf', width='stretch')
        st.button('Upload another scan', on_click=navigate, args=('Upload CT',), width='stretch')


def profile(doctor):
    heading('My profile', 'Your PanVision account information.')
    with st.container(border=True):
        st.write(f'**Display name:** {doctor["name"]}')
        st.write(f'**Email:** {doctor.get("email", "Unavailable")}')
        st.write(f'**Role:** {doctor.get("role", "doctor").title()}')
        st.caption('Passwords are never displayed. Contact your administrator for name or email changes.')
    with st.form('change_own_password', clear_on_submit=True):
        st.subheader('Change password')
        current = st.text_input('Current password', type='password', max_chars=1024)
        new = st.text_input('New password', type='password', max_chars=1024)
        confirm = st.text_input('Confirm new password', type='password', max_chars=1024)
        st.caption('Use at least 12 characters. A successful change signs you out on all devices.')
        if st.form_submit_button('Update password', type='primary'):
            if new != confirm:
                st.error('New passwords do not match.')
            elif len(new) < 12 or new == current:
                st.error('Choose a different password of at least 12 characters.')
            else:
                request('POST', '/auth/password', json={'current_password':current,'new_password':new})
                st.session_state.clear()
                st.session_state['login_notice'] = 'Password changed. Sign in with your new password.'
                st.rerun()


def scan_history(admin=False):
    scope = 'admin' if admin else 'doctor'
    params = {'limit':50}
    if admin:
        accounts = request('GET','/admin/accounts').json()
        lookup = {row['id']:row for row in accounts}
        chosen = st.selectbox('Uploader', [None]+list(lookup),
                              format_func=lambda value: 'All accounts' if value is None else f'{lookup[value]["name"]} · {lookup[value]["email"]}')
        if chosen is not None:
            params['doctor_id'] = chosen
    else:
        chosen = None
    page = st.number_input('History page', min_value=1, value=1, step=1, key=f'history_page_{scope}_{chosen}')
    params['offset'] = (page-1)*50
    result = request('GET','/admin/uploads' if admin else '/me/uploads', params=params).json()
    rows = result['items']
    st.caption(f'{result["total"]} recorded uploads · Page {page} · Timestamps shown in UTC')
    if not rows:
        st.info('No uploads on this page. Upload a scan to start a history, or return to page 1.')
        return
    a,b = st.columns([3,1])
    query = a.text_input('Search this page', placeholder='Filename or uploader…',key=f'history_search_{scope}')
    status = b.selectbox('Validation', ['All','validated','rejected'],key=f'history_status_{scope}')
    filtered = [row for row in rows if (status=='All' or row['status']==status)
                and query.lower() in (row['filename']+' '+row['doctor_name']+' '+row['doctor_email']).lower()]
    table = [{'Scan ID':row['id'],'Filename':row['filename'],'Uploader':row['doctor_name'],
              'Email':row['doctor_email'],'Status':row['status'],'Uploaded (UTC)':row['created_at'],
              'PDFs generated':row['reports']} for row in filtered]
    st.dataframe(table,hide_index=True,width='stretch')
    if table:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer,fieldnames=list(table[0]))
        writer.writeheader()
        # Keep spreadsheet apps from interpreting user filenames as formulas.
        for row in table:
            writer.writerow({k:("'"+str(v) if str(v).lstrip().startswith(('=','+','-','@')) else v) for k,v in row.items()})
        st.download_button('Export this page · CSV',buffer.getvalue(),'panvision-upload-history.csv','text/csv')
        lookup = {row['id']:row for row in filtered}
        selected = st.selectbox('Scan activity',list(lookup),format_func=lambda value: f'{lookup[value]["filename"]} · {value}')
        detail = request('GET',f'/uploads/{selected}').json()
        st.dataframe(detail['events'],hide_index=True,width='stretch')
        if 'notes' in detail and detail['notes']:
            with st.expander('Saved review notes'):
                st.text(detail['notes'])
    st.caption('History stores metadata and events, not scan bytes. Re-upload the original CT to review images again. '
               'A PDF event confirms generation, not that it was downloaded.')


def administration():
    heading('Administration', 'Manage doctor access and monitor application tables.')
    st.caption('This manages PanVision application data. PostgreSQL server administration stays with your database provider.')
    notice = st.session_state.pop('admin_notice', None)
    if notice:
        st.success(notice)
    accounts_tab, create_tab, history_tab, database_tab = st.tabs(['Accounts', 'Add doctor', 'Upload history', 'Database & audit'])
    with accounts_tab:
        accounts = request('GET', '/admin/accounts').json()
        st.dataframe([{k: row[k] for k in ('id','name','email','role','active')} for row in accounts],
                     hide_index=True, width='stretch')
        if accounts:
            lookup = {row['id']:row for row in accounts}
            selected = st.selectbox('Manage account', list(lookup),
                                    format_func=lambda value: f'{lookup[value]["name"]} · {lookup[value]["email"]}')
            account = lookup[selected]
            with st.form(f'account_access_{selected}'):
                role = st.selectbox('Access role', ['doctor','admin'], index=0 if account['role']=='doctor' else 1)
                active = st.checkbox('Account enabled', value=bool(account['active']))
                confirmed = st.checkbox('I confirm this access change')
                if st.form_submit_button('Update access', type='primary'):
                    if not confirmed:
                        st.warning('Confirm the access change first.')
                    else:
                        request('PATCH', f'/admin/accounts/{selected}', json={'active':active,'role':role})
                        st.session_state['admin_notice'] = 'Account access updated.'
                        st.rerun()
            st.caption('Disabling an account or changing its role revokes its sessions. The last active administrator is protected.')
    with create_tab:
        with st.form('admin_new_doctor', clear_on_submit=True):
            st.subheader('Create doctor account')
            name = st.text_input('Doctor name', max_chars=100)
            email = st.text_input('Doctor email', max_chars=254)
            password = st.text_input('Initial password', type='password', max_chars=1024)
            confirm = st.text_input('Confirm initial password', type='password', max_chars=1024)
            if st.form_submit_button('Create doctor', type='primary'):
                if password != confirm:
                    st.error('Passwords do not match.')
                elif not name.strip() or len(password) < 12:
                    st.error('Enter a name and a password of at least 12 characters.')
                else:
                    request('POST', '/admin/doctors', json={'name':name.strip(),'email':email,'password':password})
                    st.session_state['admin_notice'] = 'Doctor account created. Share the credentials privately.'
                    st.rerun()
        st.caption('New accounts receive doctor access. Only administrators can create accounts in the app.')
    with history_tab:
        scan_history(admin=True)
    with database_tab:
        status = request('GET', '/admin/database').json()
        for column, label, field in zip(st.columns(3), ['Accounts','Saved research cases','Active sessions'],
                                       ['accounts','saved_cases','active_sessions']):
            column.metric(label,status['counts'][field])
        st.write('**Application tables:** ' + ', '.join(status['tables']))
        st.subheader('Recent administrator actions')
        if status['audit']:
            st.dataframe(status['audit'], hide_index=True, width='stretch')
        else:
            st.info('No administrator actions recorded yet.')
        st.caption('Audit entries contain action metadata, not passwords, CT files, or report notes.')


def main():
    st.set_page_config(page_title='PanVision · Doctor Workspace', page_icon='◒', layout='wide')
    st.markdown(f'<style>{(Path(__file__).parent / "style.css").read_text()}</style>', unsafe_allow_html=True)
    doctor = require_doctor()
    idle_watchdog()
    pages = ['Home', 'Upload CT', 'Past Cases', 'Similar Cases', 'Case Report', 'My Profile']
    if doctor.get('role') == 'admin':
        pages.append('Administration')
    pending = st.session_state.pop('_pending_workspace_page', None)
    if pending in pages:
        st.session_state['workspace_page'] = pending
    if st.session_state.get('workspace_page') not in pages:
        st.session_state['workspace_page'] = 'Home'
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-icon">◒</span> PanVision</div>'
                    '<div class="brand-sub">PANCREATIC CT RESEARCH</div><div class="nav-label">WORKSPACE</div>', unsafe_allow_html=True)
        page = st.radio('Navigation', pages, key='workspace_page', label_visibility='collapsed')
        st.write(f'Signed in: {doctor["name"]}')
        if st.button('Sign out', width='stretch'):
            try:
                sign_out()
            except APIError as exc:
                st.error(str(exc))
            else:
                st.rerun()
        st.caption('Automatic sign-out after 20 minutes of inactivity.')
        st.caption('Research prototype · Not for clinical use')
    st.markdown('<div class="workspace-topbar"><span>PanVision / Doctor workspace</span>'
                f'<span>{html.escape(doctor["name"])} · Research access</span></div>', unsafe_allow_html=True)
    try:
        if page == 'Home':
            home(doctor)
        elif page == 'Upload CT':
            upload_ct()
        elif page == 'My Profile':
            profile(doctor)
        elif page == 'Administration' and doctor.get('role') == 'admin':
            administration()
        elif page == 'Past Cases':
            heading('My scan history', 'Your uploads, validation results, notes and report activity.')
            scan_history()
        elif page == 'Similar Cases':
            similar_cases()
        else:
            uploaded_report()
    except SessionExpired:
        st.session_state.clear()
        st.rerun()
    except APIError as exc:
        st.error(str(exc))
    st.markdown('<div class="footer">PanVision · Research review only. Not for clinical diagnosis.</div>', unsafe_allow_html=True)

