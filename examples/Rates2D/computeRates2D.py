import examples.Rates2D.ASSEMBLY as assembly
import examples.Rates2D.SOLVE as solve
import examples.Rates2D.L2_rates as l2rates

if __name__ == "__main__":
    num_fem_sols = 3
    assembly.main(num_fem_sols)
    solve.main(num_fem_sols)
    l2rates.main(num_fem_sols)

