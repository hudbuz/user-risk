import grpc


def raise_internal_grpc_error(context, e: Exception):
    context.set_code(grpc.StatusCode.INTERNAL)
    context.set_details(str(e))
    raise e
