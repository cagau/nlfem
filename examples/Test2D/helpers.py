import datetime
def timestamp():
    """
    Returns current timestamp as string.

    :return: string, format %m%d_%H-%M-%S
    """
    # Link to strftime Doc
    # http://strftime.org/
    return datetime.datetime.now().strftime("%m%d_%H-%M-%S")

def append_output(data, conf, kernel, load, fileHandle):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    # Write Table
    fileHandle.write("### Setting\n")
    columns = {
        "Right hand side": load["function"],
        "**Kernel**": "**"+kernel["function"]+"**",
        "**Integration Method**": "**"+conf["approxBalls"]["method"]+"**",
        "With caps": conf["approxBalls"]["isPlacePointOnCap"],
        "Quadrule outer": len(conf["quadrature"]["outer"]["weights"]),
        "Quadrule inner": len(conf["quadrature"]["inner"]["weights"]),
        "Singular quad degree": conf["quadrature"]["tensorGaussDegree"],
        "Delta": kernel["horizon"],
        "Ansatz": conf["ansatz"]
    }
    write_dict(fileHandle, columns)

    fileHandle.write("### Rates\n")
    columns = {
        "h":  data.get("h", []),
        "dof": data.get("nV_Omega", []),
        "L2 Error": data.get("L2 Error", []),
        "Rates": data.get("Rates", []),
        "Time [s]": data.get("Assembly Time", []),
    }

    write_columns(fileHandle, columns)
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