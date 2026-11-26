import unittest
from backend.web_v2.workspace_listing import page_workspace_items

class WorkspaceListingTest(unittest.TestCase):
    def setUp(self):
        self.items=[dict(id=str(i),display_name=f'protein-{i:03}.csv',storage_path=f'uploads/tool/{i}.csv',source='tool' if i<55 else 'workspace',category='table_or_text',bucket='tool_upload' if i<55 else 'user_upload',created_at=f'2026-10-{1+i%28:02}T00:00:00+00:00',size=i+1) for i in range(75)]
    def test_all_pages_reachable(self):
        a=page_workspace_items(self.items,page=1);b=page_workspace_items(self.items,page=2)
        self.assertEqual(a['total'],75);self.assertEqual(len(a['items']),50);self.assertEqual(len(b['items']),25)
        self.assertEqual(len({i['id'] for i in a['items']+b['items']}),75)
    def test_location_filters_before_pagination(self):
        result=page_workspace_items(self.items,bucket='user_upload')
        self.assertEqual(result['total'],20);self.assertTrue(all(i['bucket']=='user_upload' for i in result['items']))
    def test_facets_do_not_disappear(self):
        result=page_workspace_items(self.items,source='tool',q='protein-000')
        self.assertEqual(result['sources'],['tool','workspace']);self.assertEqual(result['total'],1)
    def test_filters_and_totals(self):
        result=page_workspace_items(self.items,modified_after='2026-10-27T00:00:00+00:00',sort='size_desc')
        self.assertTrue(all(i['created_at']>='2026-10-27T00:00:00+00:00' for i in result['items']))
        self.assertEqual(result['total_size'],sum(i['size'] for i in result['items']))
        self.assertEqual(result['items'][0]['size'],56)
    def test_out_of_range_page_recovers(self):
        result=page_workspace_items(self.items,q='protein-074',page=9)
        self.assertEqual(result['page'],1);self.assertEqual(result['items'][0]['id'],'74')
    def test_empty_result(self):
        result=page_workspace_items(self.items,file_type='structure',page=8)
        self.assertEqual((result['total'],result['page'],result['items']),(0,1,[]))

if __name__=='__main__':unittest.main()
