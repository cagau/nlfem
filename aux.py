import datetime
#from time import strftime

def timestamp():
    """
    Returns current timestamp as string.

    :return: string, format %m%d_%H_%M%S
    """
    # Link to strftime Doc
    # http://strftime.org/
    return datetime.datetime.now().strftime("%m%d_%H_%M%S")