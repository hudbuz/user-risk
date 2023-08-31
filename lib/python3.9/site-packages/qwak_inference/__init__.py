from qwak_inference.realtime_client import RealTimeClient

clients = ["RealTimeClient"]
try:
    from qwak.inner import wire_dependencies

    from qwak_inference.batch_client.batch_client import BatchInferenceClient
    from qwak_inference.feedback_client import FeedbackClient

    clients.extend(["BatchInferenceClient", "FeedbackClient"])

    wire_dependencies()
except ImportError:
    # We are conditional loading these clients since the skinny does
    # not support them due to the pandas, numpy, joblib, etc. dependencies
    print(
        "Notice that BatchInferenceClient and FeedbackClient are not available in the skinny package. "
        'In order to use them, please install them as extras: pip install "qwak-inference[batch,feedback]".'
    )

__all__ = clients

__version__ = "0.1.8"
