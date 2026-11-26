import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web_v2.model_platform import Platform, read
from backend.web_v2 import model_platform_api as api
from backend.web_v2.job_access import cookie_name


class ModelPlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.p = Platform(self.temp.name)
        self.owner = self.p.session(None)[1]
        self.other = self.p.session(None)[1]
        self.data = self.p.upload(self.owner, 'experiment.csv', ('x,group,target\n' + '\n'.join(f'{i},{i//4},{2*i+3}' for i in range(80))).encode())
        self.spec = dict(dataset_id=self.data['id'], name='Test activity', description='test fixture', category='function', target='target', features=['x'],
                         sequence_column='', group_column='group', problem='regression', method='linear')
        self.p._cache = (self.p.official(), [])
        import time
        self.p._cache_time = time.monotonic()

    def tearDown(self):
        self.temp.cleanup()

    def train(self):
        j = self.p.start(self.owner, 'train', self.spec, launch=False)
        self.p.execute(j['id'])
        result = self.p.job(self.owner, j['id'])
        self.assertEqual(result['status'], 'completed', result)
        return result['result']['model']

    def test_train_private_email_publication_and_revoke(self):
        m = self.train()
        self.assertEqual(m['split']['kind'], 'group')
        self.assertLess(m['metrics']['MAE'], 3)
        self.assertEqual(m['visibility'], 'private')
        with self.assertRaises(FileNotFoundError): self.p.model(self.other, m['id'])
        with self.assertRaises(FileNotFoundError): self.p.visibility(self.other, m['id'][5:], 'public')
        with self.assertRaisesRegex(ValueError, 'operator'):
            self.p.visibility(self.owner, m['id'][5:], 'public')
        with self.assertRaises(FileNotFoundError): self.p.model(self.other, m['id'])
        self.assertNotIn(m['id'], [x['id'] for x in self.p.catalog(self.other, query='Test activity')['items']])
        d = self.p.upload(self.owner, 'new.csv', b'x\n100\n101')
        output = self.p.predict(self.owner, m['id'], self.p.owned('datasets', d['id'], self.owner))
        self.assertEqual(len(output['rows']), 2)
        self.assertLess(abs(output['rows'][0]['target_prediction'] - 203), 4)
        self.p.visibility(self.owner, m['id'][5:], 'private')
        with self.assertRaises(FileNotFoundError): self.p.predict(self.other, m['id'], {'kind':'table','rows':[{'x':2}]})


    def test_failure_is_not_completion(self):
        m=self.train()
        bad=self.p.upload(self.owner,'bad.csv',b'other\nno\nvalues')
        j=self.p.start(self.owner,'predict',{'model_id':m['id'],'dataset_id':bad['id']},launch=False)
        self.p.execute(j['id'])
        self.assertEqual(self.p.job(self.owner,j['id'])['status'],'failed')

    def test_descriptors_chain(self):
        d=self.p.upload(self.owner,'protein.fasta',b'>a\nACDEFGHIKLMNPQRSTVWY\n>b\nACDEFGHIKL')
        model=next(m for m in self.p.official() if m['adapter']=='descriptors')
        output=self.p.predict(self.owner,model['id'],self.p.owned('datasets',d['id'],self.owner))
        self.assertEqual(output['rows'][0]['length'],20)
        self.assertAlmostEqual(sum(output['rows'][0]['fraction_'+aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'),1)

    def test_upload_and_leakage_guards(self):
        with self.assertRaises(FileNotFoundError): self.p.owned('datasets',self.data['id'],self.other)
        with self.assertRaises(ValueError): self.p.path('datasets','../anything')
        with self.assertRaises(ValueError): self.p.validate_training(self.owner,{**self.spec,'features':['x','target']})
        with self.assertRaises(ValueError): self.p.upload(self.owner,'bad.csv',b'x,x\n1,2')
        with self.assertRaises(ValueError): self.p.upload(self.owner,'model.pkl',b'weights')
        with self.assertRaises(ValueError): self.p.upload(self.owner,'bad.json',b'[{"x":1},{"y":2}]')

    def test_http_cookie_ownership_and_csrf(self):
        app=FastAPI();app.include_router(api.router)
        with patch.object(api,'service',self.p):
            with TestClient(app) as a, TestClient(app) as b:
                self.assertEqual(a.get('/api/model-platform/session').status_code,200)
                self.assertEqual(b.get('/api/model-platform/session').status_code,200)
                self.assertNotEqual(a.cookies.get(cookie_name('platform')),b.cookies.get(cookie_name('platform')))
                response=a.post('/api/model-platform/datasets',files={'file':('input.csv',b'x,y\n1,2')})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(len(b.get('/api/model-platform/datasets').json()),0)
                self.assertEqual(a.post('/api/model-platform/train',json={},headers={'origin':'https://elsewhere.example'}).status_code,403)
                model=self.train()
                self.assertEqual(b.get('/api/model-platform/models/'+model['id']).status_code,404)
                self.assertEqual(b.patch('/api/model-platform/models/'+model['id']+'/visibility',json={'visibility':'public'}).status_code,404)


if __name__=='__main__': unittest.main()
