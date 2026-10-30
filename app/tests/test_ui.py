"""Upload-first UI workflows without database or shared-drive access."""
from pathlib import Path
from types import SimpleNamespace
import io
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1]/'app.py')


class WorkspaceUITests(unittest.TestCase):
    def setUp(self):
        watchdog = patch('app.dashboard.idle_watchdog')
        watchdog.start()
        self.addCleanup(watchdog.stop)

    def test_home_and_report_gate_do_not_scan_dataset(self):
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.data.case_index') as index:
            app = AppTest.from_file(APP).run()
            self.assertEqual(app.sidebar.radio[0].value, 'Home')
            self.assertFalse(any('DATASET DIAGNOSIS' in m.value for m in app.markdown))
            app.sidebar.radio[0].set_value('Case Report').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('No scan uploaded yet' in m.value for m in app.markdown))
            next(b for b in app.button if b.label == 'Upload CT scan').click().run()
            self.assertEqual(app.sidebar.radio[0].value, 'Upload CT')
            self.assertFalse(any(b.label == 'Validate and open report' for b in app.button))
            index.assert_not_called()

    def test_uploaded_report_preview_and_pdf(self):
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request', return_value=SimpleNamespace(content=b'%PDF-test')) as api:
            app = AppTest.from_file(APP).run()
            app.session_state['upload_review'] = {'digest':'digest', 'name':'scan.nii.gz', 'content':b'ct',
                                                   'array':np.ones((4,8,8)), 'spacing':(1,1,2)}
            app.sidebar.radio[0].set_value('Case Report').run()
            self.assertFalse(app.exception)
            self.assertEqual(list(app.tabs), [])
            self.assertTrue(any('File validation</span><strong>Passed' in item.value and
                                'Model analysis</span><strong>Not run' in item.value
                                for item in app.markdown))
            app.text_area[0].set_value('Review this uploaded scan').run()
            next(b for b in app.button if b.label == 'Prepare PDF').click().run()
            self.assertEqual(app.session_state['uploaded_report_pdf'][1], b'%PDF-test')
            self.assertEqual(api.call_args.args[:2], ('POST','/uploads/report.pdf'))
            self.assertEqual(api.call_args.kwargs['data']['notes'], 'Review this uploaded scan')
            self.assertTrue(any('Start analysis when you are ready' in info.value for info in app.info))

    def test_pdf_sends_pancreas_but_excludes_lesion_mask(self):
        pancreas = np.zeros((4, 8, 8), dtype=bool)
        pancreas[1, 2:5, 3:6] = True
        lesion = np.ones((4, 8, 8), dtype=bool)
        with patch('app.dashboard.require_doctor', return_value={'id': 1, 'name': 'ABC'}), \
             patch('app.dashboard.request', return_value=SimpleNamespace(content=b'%PDF-test')) as api:
            app = AppTest.from_file(APP).run()
            app.session_state['upload_review'] = {
                'digest': 'digest', 'name': 'scan.nii.gz', 'content': b'ct',
                'array': np.ones((4, 8, 8)), 'spacing': (1, 1, 2), 'upload_id': 'upload-1',
                'pancreas_mask': pancreas, 'lesion_mask': lesion,
            }
            app.sidebar.radio[0].set_value('Case Report').run()
            next(button for button in app.button if button.label == 'Prepare PDF').click().run()
            files = api.call_args.kwargs['files']
            self.assertEqual(set(files), {'file', 'pancreas_mask'})
            np.testing.assert_array_equal(
                np.unpackbits(np.frombuffer(files['pancreas_mask'][1], dtype=np.uint8), count=pancreas.size).reshape(pancreas.shape),
                pancreas,
            )

    def test_review_notes_show_when_changes_are_unsaved(self):
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request', return_value=SimpleNamespace(content=b'%PDF-test')) as api:
            app = AppTest.from_file(APP).run()
            app.session_state['upload_review'] = {
                'digest':'digest', 'name':'scan.nii.gz', 'content':b'ct',
                'array':np.ones((4,8,8)), 'spacing':(1,1,2), 'upload_id':'upload-1',
            }
            app.sidebar.radio[0].set_value('Case Report').run()
            app.text_area[0].set_value('First note').run()
            self.assertTrue(any('Unsaved changes' in item.value for item in app.warning))

            next(button for button in app.button if button.label == 'Save review notes').click().run()
            api.assert_called_with('PUT', '/uploads/upload-1/notes', json={'notes':'First note'})
            self.assertFalse(any('Unsaved changes' in item.value for item in app.warning))
            self.assertTrue(any('Notes saved' in item.value for item in app.caption))

            app.text_area[0].set_value('Revised note').run()
            self.assertTrue(any('Unsaved changes' in item.value for item in app.warning))
            next(button for button in app.button if button.label == 'Prepare PDF').click().run()
            self.assertFalse(any('Unsaved changes' in item.value for item in app.warning))
            self.assertTrue(any('Notes saved' in item.value for item in app.caption))

    def test_successful_login_opens_home_without_form(self):
        doctor = {'id':1,'name':'ABC','email':'test@example.com'}
        def response(method, path, **kwargs):
            data = {'access_token':'token','doctor':doctor} if path == '/auth/login' else doctor
            return SimpleNamespace(json=lambda:data)
        with patch('app.auth.request', side_effect=response):
            app=AppTest.from_file(APP).run()
            app.text_input[0].set_value('test@example.com')
            app.text_input[1].set_value('test-password')
            next(b for b in app.button if b.label == 'Sign in').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.sidebar.radio[0].value,'Home')
            self.assertFalse(any(b.label == 'Sign in' for b in app.button))

    def test_upload_validates_and_waits_for_explicit_analysis(self):
        uploaded = SimpleNamespace(name='scan.nii.gz', getvalue=lambda: b'ct-test')
        output = io.BytesIO()
        np.savez_compressed(
            output, pancreas=np.ones((4, 8, 8), dtype=np.uint8),
            lesion=np.ones((4, 8, 8), dtype=np.uint8),
            gradcam=np.ones((8, 8), dtype=np.float32), gradcam_slice=np.asarray(2),
            predicted_class=np.asarray('PDAC'), pdac_probability=np.asarray(0.82),
            threshold=np.asarray(0.42), neighbor_ids=np.asarray(['101867']),
            neighbor_labels=np.asarray([1]), neighbor_similarities=np.asarray([0.96]),
        )
        def response(method, path, **kwargs):
            if path == '/uploads/validate':
                return SimpleNamespace(json=lambda:{'valid':True, 'upload_id':'upload-1'})
            if path == '/uploads/upload-1/analysis':
                return SimpleNamespace(content=output.getvalue())
            raise AssertionError(path)
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request', side_effect=response) as api, \
             patch('app.dashboard.st.file_uploader', return_value=uploaded), \
             patch('app.dashboard.read_ct', return_value=(np.ones((4,8,8)), (1,1,2))):
            app=AppTest.from_file(APP).run()
            app.sidebar.radio[0].set_value('Upload CT').run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(app.sidebar.radio[0].value, 'Case Report')
            self.assertEqual(app.session_state['upload_review']['name'], 'scan.nii.gz')
            self.assertNotIn('predicted_class', app.session_state['upload_review'])
            self.assertEqual(api.call_count, 1)
            next(button for button in app.button if button.label == 'Run full AI analysis').click().run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(app.session_state['upload_review']['predicted_class'], 'PDAC')
            self.assertTrue(app.session_state['upload_review']['lesion_mask'].all())
            self.assertEqual(api.call_count, 2)

    def test_report_has_one_explicit_analysis_button_and_no_model_tabs(self):
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request') as api:
            app=AppTest.from_file(APP).run()
            app.session_state['upload_review'] = {
                'digest':'digest', 'name':'scan.nii.gz', 'content':b'ct',
                'array':np.ones((4,8,8)), 'spacing':(1,1,2), 'upload_id':'upload-1',
            }
            app.sidebar.radio[0].set_value('Case Report').run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(list(app.tabs), [])
            self.assertEqual(sum(button.label == 'Run full AI analysis' for button in app.button), 1)
            api.assert_not_called()


    def test_doctor_profile_has_no_admin_navigation(self):
        doctor={'id':1,'name':'ABC','email':'abc@example.com','role':'doctor'}
        with patch('app.dashboard.require_doctor',return_value=doctor):
            app=AppTest.from_file(APP).run()
            self.assertNotIn('Administration',app.sidebar.radio[0].options)
            app.sidebar.radio[0].set_value('My Profile').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('abc@example.com' in m.value for m in app.markdown))

    def test_administrator_dashboard(self):
        doctor={'id':1,'name':'Admin','email':'admin@example.com','role':'admin'}
        def response(method,path,**kwargs):
            data=[{**doctor,'active':1}] if path=='/admin/accounts' else {
                'tables':['doctors','doctor_sessions','doctor_cases','admin_audit'],
                'counts':{'accounts':1,'saved_cases':0,'active_sessions':1},'audit':[]}
            if path=='/admin/uploads':
                data={'items':[],'total':0}
            return SimpleNamespace(json=lambda:data)
        with patch('app.dashboard.require_doctor',return_value=doctor),patch('app.dashboard.request',side_effect=response):
            app=AppTest.from_file(APP).run()
            self.assertIn('Administration',app.sidebar.radio[0].options)
            app.sidebar.radio[0].set_value('Administration').run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertTrue(any(b.label=='Create doctor' for b in app.button))


    def test_profile_password_change_clears_sessions(self):
        doctor={'id':1,'name':'ABC','email':'abc@example.com','role':'doctor'}
        with patch('app.dashboard.require_doctor',return_value=doctor),patch('app.dashboard.request') as api:
            app=AppTest.from_file(APP).run()
            app.session_state['api_token']='token'
            app.sidebar.radio[0].set_value('My Profile').run()
            app.text_input[0].set_value('old-password-123')
            app.text_input[1].set_value('new-password-123')
            app.text_input[2].set_value('new-password-123')
            next(b for b in app.button if b.label=='Update password').click().run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            api.assert_called_once_with('POST','/auth/password',json={'current_password':'old-password-123','new_password':'new-password-123'})
            self.assertNotIn('api_token',app.session_state)

    def test_doctor_history_and_activity(self):
        doctor={'id':1,'name':'ABC','email':'abc@example.com','role':'doctor'}
        row={'id':'11111111-1111-4111-8111-111111111111','filename':'scan.nii.gz',
             'doctor_name':'ABC','doctor_email':'abc@example.com','status':'validated',
             'created_at':'2026-10-19T00:00:00Z','reports':1}
        def response(method,path,**kwargs):
            data={'items':[row],'total':1} if path=='/me/uploads' else {
                'events':[{'event':'upload_validated','actor_id':1,'created_at':row['created_at']}], 'notes':'Saved note'}
            return SimpleNamespace(json=lambda:data)
        with patch('app.dashboard.require_doctor',return_value=doctor),patch('app.dashboard.request',side_effect=response):
            app=AppTest.from_file(APP).run()
            app.sidebar.radio[0].set_value('Past Cases').run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertEqual(len(app.dataframe),2)


    def test_similar_cases_page_states(self):
        doctor={'id':1,'name':'ABC','email':'abc@example.com','role':'doctor'}
        scan={'digest':'digest','name':'scan.nii.gz','content':b'ct','array':np.ones((4,8,8)),'spacing':(1,1,2)}
        with patch('app.dashboard.require_doctor',return_value=doctor),patch('app.dashboard.request') as api:
            app=AppTest.from_file(APP).run()
            app.sidebar.radio[0].set_value('Similar Cases').run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertTrue(any('No scan uploaded yet' in m.value for m in app.markdown))
            app.session_state['upload_review']=dict(scan)
            app.run()
            self.assertTrue(any('Analysis not run' in m.value for m in app.markdown))
            app.session_state['upload_review']=dict(scan,predicted_class='PDAC',pdac_probability=0.82,
                classification_threshold=0.5,similar_cases=[
                    {'patient_id':'100001','label':1,'similarity':0.9512},
                    {'patient_id':'100002','label':0,'similarity':0.9103},
                    {'patient_id':'100003','label':1,'similarity':0.8877}])
            app.run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertTrue(any('2 of 3' in m.value for m in app.markdown))
            table=app.dataframe[0].value
            self.assertEqual(list(table['Patient ID']),['100001','100002','100003'])
            self.assertEqual(list(table['Dataset label']),['PDAC','Non-PDAC','PDAC'])
            api.assert_not_called()
