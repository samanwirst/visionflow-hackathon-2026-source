"""Recorded real-model failure and faithful operator controls; no API calls."""
import copy
import pytest
from app.live import planner

RECORDED_PROMPT = 'Send me a local alert whenever a car comes into the whole image. Keep exactly 2.5 seconds between notifications.'
RECORDED_PROPOSAL = {'supported':True,'name':'Car presence alert in full frame',
 'explanation':'Alert when a car is present anywhere in the full frame, with a cooldown of 2.5 seconds between notifications.',
 'object_class':'car','zone':{'type':'full_frame','polygon':[]},'trigger':'presence_in_zone',
 'duration_seconds':0,'threshold':0,'cooldown_seconds':2.5,'action':'notify','focus':False,
 'visual_observation':{'question':'','labels':[],'temporal':False},'clarification_needed':'','unsupported_reason':''}

def run(monkeypatch, text, trigger, duration=0, cooldown=2.5):
    p=copy.deepcopy(RECORDED_PROPOSAL)
    p.update(trigger=trigger,duration_seconds=duration,cooldown_seconds=cooldown)
    monkeypatch.setattr(planner,'transport',lambda *_:planner.PlannerResult(status='ok',proposal=p))
    monkeypatch.setattr(planner.config,'PLANNER_COOLDOWN_SECONDS',0)
    monkeypatch.setattr(planner.config,'PLANNER_ROOM_BUDGET',100)
    return planner.plan(text,'transition-fidelity-'+text)

def test_recorded_genuine_model_presence_substitution_is_rejected(monkeypatch):
    result=run(monkeypatch,RECORDED_PROMPT,'presence_in_zone')
    assert result.status=='unsupported' and result.task is None
    assert result.proposal['trigger']=='presence_in_zone'  # No silent repair.

@pytest.mark.parametrize('text,trigger,duration',[
 (RECORDED_PROMPT,'entered_zone',0),
 ('Notify whenever a car departs from the whole image.','exited_zone',0),
 ('Notify when a car remains in the whole image for precisely 1.25 seconds.','dwell_in_zone',1.25),
 ('Watch a car currently visible in the whole image.','presence_in_zone',0),
])
def test_faithful_supported_controls(monkeypatch,text,trigger,duration):
    result=run(monkeypatch,text,trigger,duration)
    assert result.status=='ok' and result.task['trigger']==trigger
    if duration:assert result.task['duration_seconds']==1.25

@pytest.mark.parametrize('text,trigger,duration,cooldown',[
 ('Notify when a car exits the image.','presence_in_zone',0,2.5),
 ('Notify when a car stays in the image for exactly 1.25 seconds.','dwell_in_zone',1,2.5),
 (RECORDED_PROMPT,'entered_zone',0,10),
 ('Notify when a car enters and leaves the image.','entered_zone',0,2.5),
 ('Notify when a car does not enter the image.','entered_zone',0,2.5),
 ('Notify when no cars enter the image.','entered_zone',0,2.5),
 ('Email me when a car enters the image.','entered_zone',0,2.5),
 ('Notify when somebody opens the visible object.','presence_in_zone',0,2.5),
 ('Notify when two people each remain for five seconds.','presence_in_zone',0,2.5),
])
def test_lost_or_composite_constraints_are_not_approved(monkeypatch,text,trigger,duration,cooldown):
    result=run(monkeypatch,text,trigger,duration,cooldown)
    assert result.status=='unsupported' and result.task is None
