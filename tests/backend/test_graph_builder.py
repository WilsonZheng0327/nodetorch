"""Tests for graph_builder helpers that don't need a live model.

Currently focuses on find_key_nodes — the shared "where are the data / loss /
prediction nodes?" walk reused by inference, backprop, and eval code. Its contract
is easy to get subtly wrong (which port feeds the loss), so it's pinned here.
"""
from engine.graph_builder import find_key_nodes


def _edge(src, sport, tgt, tport):
    return {"source": {"nodeId": src, "portId": sport}, "target": {"nodeId": tgt, "portId": tport}}


class TestFindKeyNodes:
    def test_full_classification_graph(self):
        """data → linear → loss, with the linear feeding the loss 'predictions' port."""
        nodes = {
            "d": {"type": "data.mnist"},
            "fc": {"type": "ml.layers.linear"},
            "loss": {"type": "ml.loss.cross_entropy"},
        }
        edges = [
            _edge("d", "out", "fc", "in"),
            _edge("fc", "out", "loss", "predictions"),
            _edge("d", "labels", "loss", "labels"),
        ]
        data_nid, loss_nid, pred_nid = find_key_nodes(nodes, edges)
        assert data_nid == "d"
        assert loss_nid == "loss"
        assert pred_nid == "fc"  # the node on the 'predictions' port, not 'labels'

    def test_prediction_is_the_predictions_port_not_labels(self):
        """pred_nid must track the 'predictions' edge even when a 'labels' edge exists."""
        nodes = {
            "d": {"type": "data.mnist"},
            "logits": {"type": "ml.layers.linear"},
            "loss": {"type": "ml.loss.cross_entropy"},
        }
        edges = [
            _edge("d", "labels", "loss", "labels"),
            _edge("logits", "out", "loss", "predictions"),
        ]
        _, _, pred_nid = find_key_nodes(nodes, edges)
        assert pred_nid == "logits"

    def test_no_loss_node(self):
        nodes = {"d": {"type": "data.mnist"}, "fc": {"type": "ml.layers.linear"}}
        data_nid, loss_nid, pred_nid = find_key_nodes(nodes, [])
        assert data_nid == "d"
        assert loss_nid is None
        assert pred_nid is None

    def test_loss_present_but_nothing_on_predictions_port(self):
        nodes = {"d": {"type": "data.mnist"}, "loss": {"type": "ml.loss.mse"}}
        edges = [_edge("d", "labels", "loss", "labels")]  # no 'predictions' edge
        data_nid, loss_nid, pred_nid = find_key_nodes(nodes, edges)
        assert (data_nid, loss_nid, pred_nid) == ("d", "loss", None)

    def test_no_data_node(self):
        nodes = {"loss": {"type": "ml.loss.mse"}}
        data_nid, loss_nid, _ = find_key_nodes(nodes, [])
        assert data_nid is None
        assert loss_nid == "loss"

    def test_recognizes_vae_and_gan_as_loss(self):
        """ALL_LOSS_NODES includes vae/gan, which lack a single scalar loss but still count."""
        for loss_type in ("ml.loss.vae", "ml.loss.gan"):
            nodes = {"d": {"type": "data.mnist"}, "L": {"type": loss_type}}
            _, loss_nid, _ = find_key_nodes(nodes, [])
            assert loss_nid == "L"
