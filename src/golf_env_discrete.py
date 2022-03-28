from src.golf_env import GolfEnv


class GolfEnvDiscrete(GolfEnv):
    def __init__(self):
        super(GolfEnvDiscrete, self).__init__()
        self.club_info = [
            # NAME          DIST    VAR_X    VAR_Y
            ('DRIVER-L',    260,     10,     10),
            ('DRIVER-S',    200,     10,     10),
            ('3WOOD-L',     235,     10,     10),
            ('3WOOD-S',     180,     10,     10),
            ('5WOOD-L',     210,     10,     10),
            ('5WOOD-S',     170,     10,     10),
            ('3IRON-L',     200,     10,     10),
            ('3IRON-S',     160,     10,     10),
            ('4IRON-L',     180,     10,     10),
            ('4IRON-S',     150,     10,     10),
            ('5IRON-L',     170,     10,     10),
            ('5IRON-S',     140,     10,     10),
            ('6IRON-L',     160,     10,     10),
            ('6IRON-S',     130,     10,     10),
            ('7IRON-L',     150,     10,     10),
            ('7IRON-S',     120,     10,     10),
            ('8IRON-L',     140,     10,     10),
            ('8IRON-S',     110,     10,     10),
            ('9IRON-L',     130,     10,     10),
            ('9IRON-S',     95,      10,     10),
            ('PW-L',        120,     10,     10),
            ('PW-S',        80,      10,     10),
            ('SW-L',        100,     10,     10),
            ('SW-S',        60,      10,     10),
        ]
        self.selected_club_info = None

    def _get_flight_model(self, distance_action):
        self.selected_club_info = self.club_info[distance_action]
        return self.selected_club_info[1:4]  # exclude name

    def _generate_debug_str(self, msg):
        return ' used' + str(self.selected_club_info) + ' ' + msg
