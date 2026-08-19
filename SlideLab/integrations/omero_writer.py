"""
omero_writer.py

Integration module for persisting SlideLab AI results, quality control metrics,
PAC (Patch Attention Certainty) scores, and ROIs directly into OMERO Server
following the "Database-First" paradigm.

Authors: ColabBio Engineering & Lorenzo Olmo Marchal
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SlideLab.OmeroWriter")

class OmeroWriter:
    """
    Client for writing SlideLab preprocessing, QC, and MIL results back to OMERO.
    """
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 4064,
        username: Optional[str] = None,
        password: Optional[str] = None,
        session_id: Optional[str] = None,
        group: Optional[str] = None
    ):
        self.host = host or os.environ.get("OMERO_HOST", "omero.colabbio.lan")
        self.port = int(port or os.environ.get("OMERO_PORT", 4064))
        self.username = username or os.environ.get("OMERO_USER", "root")
        self.password = password or os.environ.get("OMERO_PASS", "")
        self.session_id = session_id or os.environ.get("OMERO_SESSION", None)
        self.group = group or os.environ.get("OMERO_GROUP", None)
        self._conn = None

    def connect(self):
        """Establish BlitzGateway connection to OMERO."""
        try:
            from omero.gateway import BlitzGateway
            if self.session_id:
                self._conn = BlitzGateway(session_uuid=self.session_id, host=self.host, port=self.port)
            else:
                self._conn = BlitzGateway(self.username, self.password, host=self.host, port=self.port)
            
            connected = self._conn.connect()
            if not connected:
                raise ConnectionError(f"Failed to connect to OMERO server at {self.host}:{self.port}")
            if self.group:
                self._conn.setGroupForSession(self.group)
            logger.info(f"Connected to OMERO at {self.host}:{self.port} as user '{self._conn.getUser().getName()}'.")
            return self._conn
        except ImportError:
            logger.warning("omero-py is not installed in the current environment. Operating in mock/offline mode.")
            return None
        except Exception as e:
            logger.error(f"Error connecting to OMERO: {e}")
            raise

    def close(self):
        """Close connection to OMERO."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("OMERO connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_map_annotation(
        self,
        image_id: int,
        key_value_pairs: Dict[str, Any],
        namespace: str = "colabbio.ai.findings"
    ) -> bool:
        """
        Attaches a MapAnnotation to a target OMERO Image.
        """
        if self._conn is None:
            logger.info(f"[Offline] MapAnnotation for Image ID {image_id} [{namespace}]: {key_value_pairs}")
            return True

        try:
            import omero.model
            from omero.rtypes import rstring

            image = self._conn.getObject("Image", image_id)
            if not image:
                logger.error(f"Image with ID {image_id} not found in OMERO.")
                return False

            map_ann = omero.model.MapAnnotationI()
            map_ann.setNs(rstring(namespace))
            
            # Format as list of pairs [[key, value], ...]
            formatted_pairs = [[str(k), str(v)] for k, v in key_value_pairs.items()]
            map_ann.setMapValue(formatted_pairs)
            
            map_ann = self._conn.getUpdateService().saveAndReturnObject(map_ann)
            self._conn.linkAnnotation(image, map_ann)
            logger.info(f"Successfully attached MapAnnotation (ID: {map_ann.getId().getValue()}) to Image {image_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to write MapAnnotation to Image {image_id}: {e}")
            return False

    def write_pac_and_qc_metrics(
        self,
        image_id: int,
        metrics: Dict[str, Any],
        pac_score: Optional[float] = None,
        prediction_label: Optional[str] = None,
        prediction_confidence: Optional[float] = None
    ) -> bool:
        """
        Standardized payload helper for SlideLab AI results.
        """
        payload = {
            "engine": "SlideLab-On-The-Fly",
            "version": "2.1.0",
            "num_tiles": metrics.get("num_tiles", 0),
            "cpu_time_sec": round(metrics.get("elapsed_cpu_seconds", 0), 2),
            "user_time_sec": round(metrics.get("elapsed_user_seconds", 0), 2),
            "feature_extractor": metrics.get("feature_extractor", "resnet50"),
            "device": metrics.get("device", "cuda"),
        }
        if pac_score is not None:
            payload["mean_pac_score"] = round(pac_score, 4)
        if prediction_label is not None:
            payload["prediction_label"] = prediction_label
        if prediction_confidence is not None:
            payload["prediction_confidence"] = round(prediction_confidence, 4)

        return self.write_map_annotation(image_id, payload, namespace="colabbio.ai.slidelab.qc_and_mil")

    def write_qc_metrics(
        self,
        image_id: Any,
        pac_score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """Alias helper for write_pac_and_qc_metrics."""
        return self.write_pac_and_qc_metrics(
            image_id=image_id,
            metrics=metrics or {},
            pac_score=pac_score,
            **kwargs
        )


def main():
    parser = argparse.ArgumentParser(description="SlideLab OMERO Write-Back Utility")
    parser.add_argument("--image_id", type=int, required=True, help="OMERO Image ID to annotate")
    parser.add_argument("--metrics_json", type=str, required=True, help="Path to SlideLab _metrics.json file")
    parser.add_argument("--pac_score", type=float, default=None, help="Optional PAC score")
    parser.add_argument("--prediction", type=str, default=None, help="Optional diagnosis prediction string")
    parser.add_argument("--confidence", type=float, default=None, help="Optional confidence (0.0 - 1.0)")
    args = parser.parse_args()

    if not os.path.isfile(args.metrics_json):
        raise FileNotFoundError(f"Metrics file {args.metrics_json} not found.")

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    writer = OmeroWriter()
    with writer:
        writer.write_pac_and_qc_metrics(
            image_id=args.image_id,
            metrics=metrics,
            pac_score=args.pac_score,
            prediction_label=args.prediction,
            prediction_confidence=args.confidence
        )

if __name__ == "__main__":
    main()
