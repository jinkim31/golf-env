from src.golf_env import GolfEnv


class GolfEnvDiscrete(GolfEnv):
    def __init__(self):
        super(GolfEnvDiscrete, self).__init__()
        self.flight_models = (
            # NAME    DIST    DEV_X   DEV_Y
            ('DR',    230,    60,     60),
            ('W3',    215,    55,     55),
            ('W5',    195,    40,     40),
            ('I3',    180,    40,     40),
            ('I4',    170,    35,     35),
            ('I5',    160,    30,     30),
            ('I6',    150,    30,     30),
            ('I7',    140,    30,     30),
            ('I8',    130,    30,     30),
            ('I9',    115,    35,     35),
            ('PW',    105,    40,     40),
            ('SW',    80,     40,     40),
            ('SW',    70,     35,     35),
            ('SW',    60,     30,     30),
            ('SW',    50,     20,     20),
            ('SW',    40,     15,     15),
            ('SW',    30,     10,     10),
            ('SW',    20,     5,      5),
            ('SW',    10,     3,      3),
            ('SW',    5,      1,      1),
        )
        self.selected_club_info = None

    def _get_flight_model(self, distance_action):
        self.selected_club_info = self.flight_models[distance_action]
        return self.selected_club_info[1:4]  # exclude name

    def _generate_debug_str(self, msg):
        return 'used' + str(self.selected_club_info) + ' ' + msg
