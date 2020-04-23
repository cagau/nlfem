import examples.Rates2D.ASSEMBLY_nonreg as assembly
import examples.Rates2D.MassMatrix as massmat
import examples.Rates2D.SOLVE_nonreg as solve
import examples.Rates2D.L2_rates_nonreg_scipyinterp as l2rates


if __name__ == "__main__":

    assembly.main()
    massmat.main()
    solve.main()
    l2rates.main()