"""Host gate tests; they provide no timing or GPU acceptance evidence."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from check_lfm3_native_policy_benchmark import require_eager


@pytest.mark.parametrize('damage',['','hash','count','distribution','numeric','reset','duplicate'])
def test_benchmark_requires_original_exact_eager_gate(damage):
    evidence=dict(binary_sha256='test-hash',cases=[dict(case=name,activation_checks=count,max_relative_rms=0.0,max_logit_abs=0.0,teacher_argmax_exact=True,greedy_exact=True,reset_exact_logits=True) for name,count in [('baseline',4240),('arithmetic',530),('window257',1060),('single',1060)]])
    if damage=='hash':evidence['binary_sha256']='wrong'
    elif damage=='count':evidence['cases'][0]['activation_checks']-=1
    elif damage=='distribution':evidence['cases'][0]['activation_checks']-=1;evidence['cases'][1]['activation_checks']+=1
    elif damage=='numeric':evidence['cases'][0]['max_logit_abs']=0.01
    elif damage=='reset':evidence['cases'][0]['reset_exact_logits']=False
    elif damage=='duplicate':evidence['cases'][1]['case']='baseline'
    if damage:
        with pytest.raises(ValueError):require_eager(evidence,'test-hash')
    else:require_eager(evidence,'test-hash')
