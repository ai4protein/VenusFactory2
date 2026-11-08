"""Offline HTTP checks for the summary route on every supported Web harness."""
import asyncio
import json

import httpx
import pytest

from agent.harness.contracts import ModelBinding, Profile, Budget
from agent.harness.gateway import EvidenceGateway
from agent.harness.messages_bridge import MessagesBridge
from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
from agent.rsi.engine.model_gateway import GatewayPolicy
from backend.web_v2.agent_context import summarize_via_relay


@pytest.mark.parametrize('harness',['kimi-code','deepseek-harness','claude-code','codex'])
def test_same_model_relay_and_shared_call_budget(tmp_path,harness):
    async def scenario():
        responses=harness=='codex'
        profile=Profile(mode='science_agent',harness=harness,
                        model=ModelBinding(provider='offline',model='test-free',base_url='https://offline.invalid/v1',
                                           protocol='responses' if responses else 'chat_completions'),
                        budget=Budget(model_calls=2,thinking=None))
        bodies=[]
        def upstream(request):
            body=json.loads(request.content);bodies.append(body)
            assert body['model']=='test-free' and not body.get('tools')
            if responses:
                assert request.url.path=='/v1/responses'
                response={'id':'resp_test','object':'response','model':'test-free','status':'completed',
                          'output':[{'type':'message','id':'msg_test','status':'completed','role':'assistant',
                                     'content':[{'type':'output_text','text':'Protein P12345; preserve /workspace/a.csv.'}]}]}
            else:
                assert request.url.path=='/v1/chat/completions'
                response={'id':'chat_test','model':'test-free','choices':[{'index':0,'message':{'role':'assistant','content':'Protein P12345; preserve /workspace/a.csv.'},'finish_reason':'stop'}]}
            return httpx.Response(200,json=response)
        if responses:
            gateway=ResponsesGateway;policy=ResponsesPolicy.from_profile(profile,[])
        else:
            gateway=MessagesBridge if harness=='claude-code' else EvidenceGateway
            policy=GatewayPolicy(model='test-free',mode='science_agent',scientific_tools=[],max_calls=2)
        async with gateway(policy=policy,upstream_url=profile.model.base_url,api_key='test-only',
                           output=tmp_path/'requests.jsonl',transport=httpx.MockTransport(upstream)) as relay:
            first=await summarize_via_relay(relay,profile,'Untrusted historical transcript',2000)
            second=await summarize_via_relay(relay,profile,'More historical transcript',2000)
            assert first==second and relay.calls==2
            with pytest.raises(httpx.HTTPStatusError) as error:
                await summarize_via_relay(relay,profile,'Third request',2000)
            assert error.value.response.status_code==400 and len(bodies)==2
    asyncio.run(scenario())
