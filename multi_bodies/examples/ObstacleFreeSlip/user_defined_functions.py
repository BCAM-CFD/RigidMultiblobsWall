'''
Example for an obstacle with free slip.
'''

import multi_bodies_functions
from multi_bodies_functions import *

def set_slip_by_ID_new(body, slip, *args, **kwargs):
  body.function_slip = partial(flow_resolved, *args, **kwargs)
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new


def flow_resolved(body, *args, **kwargs):
  '''
  It adds the constant background flow.
  '''
  flow = np.zeros((b.Nblobs, 3))
  flow[:,2] = 1.0
  return  flow

