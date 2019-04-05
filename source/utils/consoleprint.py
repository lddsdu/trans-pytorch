import time


def ctime():
    c = time.time()
    strtime = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(c))
    return strtime


def __consoleprint(mode, info):
    return "{} {} {}".format(ctime(), mode, info)


def consoleinfo(info):
    print(__consoleprint("INFO", info))


def consolewarn(info):
    print(__consoleprint("WARN", info))


if __name__ == '__main__':
    consoleinfo("aaa")
    consolewarn("bbb")
