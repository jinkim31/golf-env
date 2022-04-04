from src.golf_env import GolfEnv


class GolfEnvDiscrete(GolfEnv):
    def __init__(self):
        super(GolfEnvDiscrete, self).__init__()
        self.flight_models = (
            # NAME    DIST    DEV_X   DEV_Y
            ('DR',    230,    15,     15),
            ('W3',    215,    15,     15),
            ('W5',    195,    15,     15),
            ('I3',    180,    15,     15),
            ('I4',    170,    15,     15),
            ('I5',    160,    15,     15),
            ('I6',    150,    15,     15),
            ('I7',    140,    15,     15),
            ('I8',    130,    15,     15),
            ('I9',    115,    15,     15),
            ('PW',    105,    15,     15),
            ('SW',    80,     15,     15),
            ('SW',    70,     15,     15),
            ('SW',    60,     15,     15),
            ('SW',    50,     15,     15),
            ('SW',    40,     15,     15),
            ('SW',    30,     15,     15),
            ('SW',    20,     15,     15),
            ('SW',    10,     15,     15),
            ('SW',    5,      15,     15),
        )
        self.selected_club_info = None

    def _get_flight_model(self, distance_action):
        self.selected_club_info = self.flight_models[distance_action]
        return self.selected_club_info[1:4]  # exclude name

    def _generate_debug_str(self, msg):
        return 'used' + str(self.selected_club_info) + ' ' + msg
