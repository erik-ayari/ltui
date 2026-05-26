import json

from ltui.lightning import LtuiLogger, StructuredMetricWriter


PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000150a0f53a0000000049454e44ae426082"
)


def test_ltui_logger_exposes_read_only_lightning_paths(tmp_path):
    logger = LtuiLogger(save_dir=tmp_path, name="stage1")

    assert logger.save_dir == str(tmp_path.resolve())
    assert logger.log_dir == tmp_path.resolve() / "stage1" / "version_0"
    assert logger.experiment.log_dir == logger.log_dir


def test_structured_metric_writer_records_manifest_and_series(tmp_path):
    writer = StructuredMetricWriter(tmp_path, train_prefix="fit", val_prefix="valid")

    writer.log_metrics({"fit/loss/kl": 1.2, "valid/loss/kl": 0.9, "epoch": 3}, step=42)

    manifest = json.loads((tmp_path / "ltui_manifest.json").read_text())
    by_name = {item["name"]: item for item in manifest["series"]}
    assert by_name["fit/loss/kl"]["role"] == "train"
    assert by_name["fit/loss/kl"]["metric_path"] == ["loss", "kl"]
    assert by_name["valid/loss/kl"]["role"] == "val"
    rows = (tmp_path / "series" / "fit" / "loss" / "kl.csv").read_text().splitlines()
    assert rows[0] == "step,epoch,wall_time,value"
    assert rows[1].split(",")[:2] == ["42.0", "3.0"]
    assert rows[1].split(",")[-1] == "1.2"


def test_structured_metric_writer_supports_metric_nodes_with_children(tmp_path):
    writer = StructuredMetricWriter(tmp_path)

    writer.log_metrics({"train/pose_head/loss": 1.0, "train/pose_head/loss/orientation_cosine_error": 0.2}, step=1)

    manifest = json.loads((tmp_path / "ltui_manifest.json").read_text())
    names = {item["name"]: item["metric_path"] for item in manifest["series"]}
    assert names == {
        "train/pose_head/loss": ["pose_head", "loss"],
        "train/pose_head/loss/orientation_cosine_error": ["pose_head", "loss", "orientation_cosine_error"],
    }


def test_structured_metric_writer_supports_prefix_style_roles(tmp_path):
    writer = StructuredMetricWriter(tmp_path)

    writer.log_metrics({"train_loss": 1.0, "val_loss/kl": 0.8}, step=2)

    manifest = json.loads((tmp_path / "ltui_manifest.json").read_text())
    by_name = {item["name"]: item for item in manifest["series"]}
    assert by_name["train_loss"]["role"] == "train"
    assert by_name["train_loss"]["metric_path"] == ["loss"]
    assert by_name["val_loss/kl"]["role"] == "val"
    assert by_name["val_loss/kl"]["metric_path"] == ["loss", "kl"]


def test_structured_metric_writer_records_images_in_step_order(tmp_path):
    writer = StructuredMetricWriter(tmp_path)

    writer.log_image("train/recon/sample", PNG_BYTES, step=10, epoch=1, wall_time=10)
    writer.log_image("train/recon/sample", PNG_BYTES, step=2, epoch=0, wall_time=2)

    manifest = json.loads((tmp_path / "ltui_manifest.json").read_text())
    image = manifest["images"][0]
    assert image["name"] == "train/recon/sample"
    assert image["role"] == "train"
    assert image["image_path"] == ["recon", "sample"]
    assert image["path"] == "images/train/recon/sample"

    names = sorted(path.name for path in (tmp_path / image["path"]).iterdir())
    assert names[0].startswith("step_000000000002")
    assert names[1].startswith("step_000000000010")


def test_structured_metric_writer_supports_custom_prefix_style_roles(tmp_path):
    writer = StructuredMetricWriter(tmp_path, train_prefix="train_", val_prefix="val_")

    writer.log_metrics({"train_loss/kl": 1.0, "val_loss/kl": 0.8}, step=2)

    manifest = json.loads((tmp_path / "ltui_manifest.json").read_text())
    by_name = {item["name"]: item for item in manifest["series"]}
    assert by_name["train_loss/kl"]["role"] == "train"
    assert by_name["train_loss/kl"]["metric_path"] == ["loss", "kl"]
    assert by_name["val_loss/kl"]["role"] == "val"
    assert by_name["val_loss/kl"]["metric_path"] == ["loss", "kl"]
