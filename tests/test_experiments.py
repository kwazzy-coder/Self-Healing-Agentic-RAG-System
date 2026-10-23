import glob
import os

import pytest

import config
from experiments.config_schema import ExperimentConfig
from experiments.runner import config_overrides, pipeline_config_for, run_experiment
from pipeline import Components
from tests.fakes import FakeChatClient, make_llm
from tests.test_evaluator import SAMPLES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "experiments", "configs", "*.yaml")))


@pytest.mark.parametrize("path", CONFIGS, ids=os.path.basename)
def test_every_config_is_valid(path):
    exp = ExperimentConfig.from_yaml(path)
    pipeline_config_for(exp)
    with config_overrides(exp.config_overrides()):
        pass


def test_overrides_apply_and_restore():
    before = config.ENABLE_NLI
    with config_overrides({"ENABLE_NLI": not before}):
        assert config.ENABLE_NLI is (not before)
    assert config.ENABLE_NLI is before
    with pytest.raises(KeyError):
        with config_overrides({"NOT_A_SETTING": 1}):
            pass


def test_run_experiment_end_to_end(tmp_path, retriever, nli):
    client = FakeChatClient(answers={"nationality": "British", "capital of France": "Paris"})
    comp = Components(llm=make_llm(client), retriever=retriever, nli=nli)
    report = run_experiment(os.path.join(ROOT, "experiments", "configs", "ablation_no_nli.yaml"),
                            output_dir=str(tmp_path), components=comp, samples_override=SAMPLES)
    assert set(report["aggregate"]) == {"vanilla_rag", "ablation_no_nli"}
    assert report["aggregate"]["ablation_no_nli"]["em"]["mean"] == 1.0
    assert os.path.exists(tmp_path / "ablation_no_nli.json")
