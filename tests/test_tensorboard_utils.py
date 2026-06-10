from argparse import Namespace

from utils.tensorboard_utils import write_training_metadata


class DummyWriter:
    def __init__(self):
        self.text = []
        self.scalars = []

    def add_text(self, tag, text_string, global_step=None):
        self.text.append((tag, text_string, global_step))

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.scalars.append((tag, scalar_value, global_step))


def test_write_training_metadata_records_config_and_optimizers():
    writer = DummyWriter()
    args = Namespace(iterations=3, gaussian_optimizer_type="muon_schedulefree")

    write_training_metadata(writer, args, {"gaussian": "schedulefree(muon)", "deform": "Adam"})

    tags = [item[0] for item in writer.text]
    assert "config/args" in tags
    assert "config/optimizers" in tags
    assert ("config/iterations", 3, 0) in writer.scalars
    assert "muon_schedulefree" in writer.text[0][1]
    assert "schedulefree(muon)" in writer.text[1][1]


def test_write_training_metadata_allows_missing_tensorboard_writer():
    args = Namespace(iterations=3)

    write_training_metadata(None, args, {"gaussian": "Adam"})
