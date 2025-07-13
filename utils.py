import datetime


def str_to_epoch(s):
    try:
        dt = datetime.datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f%z')
        except ValueError:
            try:
                dt = datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S%z')
            except ValueError:
                try:
                    dt = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f%z')
                except ValueError:
                    try:
                        dt = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S%z')
                    except ValueError:
                        try:
                            dt = datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f')
                        except ValueError:
                            try:
                                dt = datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
                            except ValueError:
                                try:
                                    dt = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
                                except ValueError:
                                    try:
                                        dt = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
                                    except ValueError:
                                        try:
                                            dt = datetime.datetime.strptime(s, '%Y-%m-%d')
                                        except ValueError:
                                            raise ValueError(f"Invalid date string: {s}")
    return int(dt.timestamp())