import sys

sys.path.append('network')

from evaluate import isolated_obj_scenario 
from evaluate import pack_scenario 
from evaluate import pile_scenario

if __name__ == '__main__':
    # isolated_obj_scenario(100, vis=False, output=True, debug=False)
    # pack_scenario(100, vis=False, output=True, debug=False)

    # baseline pile scenario
    # pile_scenario(100, vis=False, output=True, debug=False, baseline=True)

    # new pile scenario
    pile_scenario(10, vis=True, show_output=True, debug=True, baseline=False)

