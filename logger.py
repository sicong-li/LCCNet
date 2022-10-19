
INFO_CONST = 1
DEBUG_CONST =2

LOGGER_LEVE = INFO_CONST

def INFO(msg):
    if LOGGER_LEVE >= INFO_CONST:
        print(msg)

def DEBUG(msg):
    if LOGGER_LEVE >= DEBUG_CONST:
        print(msg)