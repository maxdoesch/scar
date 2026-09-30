import omegaconf

from mme_vla_suite.models.config.history_config_utils import save_history_config


def test_save_history_config_writes_yaml_for_dictconfig(tmp_path):
    history_config = omegaconf.OmegaConf.create({"query_reg_len": 2, "mem_sink_len": 0})

    save_history_config(history_config, tmp_path)

    output_path = tmp_path / "history_config.yaml"
    assert output_path.exists()
    loaded = omegaconf.OmegaConf.load(output_path)
    assert loaded.query_reg_len == 2
    assert loaded.mem_sink_len == 0


def test_save_history_config_writes_yaml_for_dict(tmp_path):
    history_config = {"query_reg_len": 1, "mask_or_pad": "mask"}

    save_history_config(history_config, tmp_path)

    output_path = tmp_path / "history_config.yaml"
    assert output_path.exists()
    loaded = omegaconf.OmegaConf.load(output_path)
    assert loaded.query_reg_len == 1
    assert loaded.mask_or_pad == "mask"
