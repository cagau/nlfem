import datetime
import os # to create an interface with our operating system
import sys # information on how our code is interacting with the host system

def timestamp():
    """
    Returns current timestamp as string.

    :return: string, format %m%d_%H-%M-%S
    """
    # Link to strftime Doc
    # http://strftime.org/
    return datetime.datetime.now().strftime("%m%d_%H-%M-%S")

def append_output(data, conf, kernel, load, fileHandle, datacolumns=None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    # Write Table
    fileHandle.write("### Setting\n")
    columns = {
        "Ansatz space": conf["ansatz"],
        "Right hand side": load["function"],
        "**Kernel**": "**"+kernel["function"]+"**",
        "Horizon $\delta$": kernel["horizon"],
        "Fractional constant $s$\n(Default -1)": kernel.get("fractional_s", -1),
        "**Intgr. remote pairs**": "**"+conf["approxBalls"]["method"]+"**",
        "With caps": conf["approxBalls"]["isPlacePointOnCap"],
        "Quadrule outer element": len(conf["quadrature"]["outer"]["weights"]),
        "Quadrule inner element": len(conf["quadrature"]["inner"]["weights"]),
        "**Intgr. close pairs**\n(Relevant only if singular)": "**"+conf.get("closeElements", conf["approxBalls"]["method"])+"**",
        "Singular quad degree": conf["quadrature"]["tensorGaussDegree"],
    }
    write_dict(fileHandle, columns)

    fileHandle.write("### Rates\n")
    if datacolumns is None:
        datacolumns = {
            "$h$":  data.get("$h$", []),
            "$K_\Omega$": data.get("$K_\Omega$", []),
            "L2 Error": data.get("L2 Error", []),
            "Rates": data.get("Rates", []),
            "Time [s]": data.get("Assembly Time", [])#
        }

    write_columns(fileHandle, datacolumns)
    fileHandle.write('\\newpage \n')
    #fileHandle.close()



def write_columns(f, columns, titles=None):
    n_cols = len(columns)

    for key, value in columns.items():
        f.write("| " + key)
    f.write("| \n")

    row=""
    for i in range(n_cols):
        row += "|---"
    row+="|"
    f.write(row+"\n")

    n_rows = [len(columns[k]) for k in columns.keys()]

    for row in range(max(n_rows)):
        col_number = 0 # ColumNumber
        for key, value in columns.items():
            if row < n_rows[col_number]:
                f.write("| " + "{0:1.2e}".format(value[row]) + " ")
            else:
                f.write("| ")
            col_number += 1
        f.write("|\n")


def write_dict(f, rows):
    f.write("| | |\n| --- | --- |\n")
    for key, value in rows.items():
        f.write("| " + key + " | " + str(value) + " |\n")