ERR_DATA_NOT_IN_DISK = ValueError(
    'ERR_DATA_NOT_IN_DISK')


def err_not_support_instrument(instrument: str):
    return ValueError('ERR_NOT_SUPPORT_INSTRUMENT', instrument)
