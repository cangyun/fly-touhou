import pytest
from fly_touhou import thrl_vector


def test_dead_worker_is_reported_without_waiting_for_unrelated_long_replay(monkeypatch):
    class Connection:
        def recv(self):
            raise EOFError('worker exited')
    vector=thrl_vector.THRLNativeVector.__new__(thrl_vector.THRLNativeVector)
    vector._connections=[Connection(),object()]
    vector.timeout=180;vector.failures=[];vector._failed=False
    vector._note=lambda *args:None
    vector._persist=lambda:None
    calls=[]
    def ready(connections,timeout):
        calls.append(1)
        if len(calls)>1:pytest.fail('Waited for the unrelated slow worker after confirmed EOF')
        return [vector._connections[0]]
    monkeypatch.setattr(thrl_vector,'wait',ready)
    with pytest.raises(thrl_vector.THRLVectorError):
        vector._gather([0,1],1,'reset')
    assert vector.failures[0]['index']==0 and vector._failed
