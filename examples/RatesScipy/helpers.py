def write_output(data):
    import conf
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import os


    # Write Plots

    pp = PdfPages(conf.fnames["timePlot.pdf"])
    plt.plot(data["h"], data["Assembly Time"], c="b")
    plt.scatter(data["h"], data["Assembly Time"], c="b")
    plt.xlabel("h")
    plt.ylabel("time [sec]")
    plt.title("Assembly Time")
    plt.savefig(pp,  format='pdf')
    plt.close()

    plt.plot(np.array(data["nV_Omega"])**2, data["Assembly Time"], c="b")
    plt.scatter(np.array(data["nV_Omega"])**2, data["Assembly Time"], c="b")
    plt.xlabel("Number of basis function in Omega squared, $nV^2$")
    plt.ylabel("time [sec]")
    plt.title("Assembly Time")
    plt.savefig(pp,  format='pdf')
    plt.close()
    pp.close()

    # Write Table

    f = open(conf.fnames["rates.md"], "w+")

    f.write("# Settings\n")
    columns = {
        "Right hand side": conf.model_f,
        "Kernel": conf.model_kernel,
        "Integration Method": conf.integration_method,
        "With caps": conf.is_PlacePointOnCap,
        "Quadrule outer": conf.quadrule_outer,
        "Quadrule inner": conf.quadrule_inner,
        "Delta": conf.delta,
        "Ansatz": conf.ansatz,
        "Boundary Conditions": conf.boundaryConditionType
    }
    write_dict(f, columns)

    f.write("# Rates\n")
    columns = {
        "h":  data.get("h", []),
        "L2 Error": data.get("L2 Error", []),
        "Rates": data.get("Rates", []),
        "Assembly Time": data.get("Assembly Time", []),
    }


    write_columns(f, columns)
    f.close()

    # Create Pdf
    os.system("pandoc " + conf.fnames["rates.md"] + " -o " + conf.fnames["rates.pdf"])
    os.system("pdfunite " + conf.fnames["rates.pdf"] + " "
              + conf.fnames["timePlot.pdf"] + " "
              + conf.fnames["report.pdf"])

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