from src.golf_env import GolfEnv


class GolfEnvDiscrete(GolfEnv):
    def __init__(self):
        super(GolfEnvDiscrete, self).__init__()
        self.flight_models = [
            # NAME          DIST    DEV_X   DEV_Y
            ('DRIVER-L',    260,    15,     15),
            ('DRIVER-S',    200,    15,     15),
            ('3WOOD-L',     235,    10,     10),
            ('3WOOD-S',     180,    17.5,   17.5),
            ('5WOOD-L',     210,    7.5,    7.5),
            ('5WOOD-S',     170,    12.5,   12.5),
            ('3IRON-L',     200,    10,     10),
            ('3IRON-S',     160,    10,     10),
            ('4IRON-L',     180,    7.5,    7.5),
            ('4IRON-S',     150,    10,     10),
            ('5IRON-L',     170,    5,      5),
            ('5IRON-S',     140,    10,     10),
            ('6IRON-L',     160,    5,      5),
            ('6IRON-S',     130,    10,     10),
            ('7IRON-L',     150,    5,      5),
            ('7IRON-S',     120,    10,     10),
            ('8IRON-L',     140,    5,      5),
            ('8IRON-S',     110,    10,     10),
            ('9IRON-L',     130,    7.5,    7.5),
            ('9IRON-S',     95,     10,     10),
            ('PW-L',        120,    7.5,    7.5),
            ('PW-S',        80,     12.5,   12.5),
            ('SW-L',        100,    10,     10),
            ('SW-S',        60,     10,     10),
        ]
        self.selected_club_info = None

    def _get_flight_model(self, distance_action):
        self.selected_club_info = self.flight_models[distance_action]
        return self.selected_club_info[1:4]  # exclude name

    def _generate_debug_str(self, msg):
        return 'used' + str(self.selected_club_info) + ' ' + msg