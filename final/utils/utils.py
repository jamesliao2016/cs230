import argparse

DEBUG = False

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def assert_shape(t, exp_shape_list):
    """Utility assertion method to check if TensorFlow shape matches expected"""
    if DEBUG:
        actual = t.get_shape().as_list()
        assert actual == exp_shape_list, '\n\texpect: {}\n\tactual: {}'.format(exp_shape_list, actual)
    else:
        pass

