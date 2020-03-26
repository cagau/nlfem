import examples.Rates2D_reg.ASSEMBLY as assembly
import examples.Rates2D_reg.SOLVE as solve
import examples.Rates2D_reg.L2_rates as l2rates
from examples.Rates2D_reg.nlocal import write_output

if __name__ == "__main__":
    num_fem_sols = 5
    data = {}

    data.update(assembly.main(num_fem_sols))
    solve.main(num_fem_sols)
    data.update(l2rates.main(num_fem_sols))
    write_output(data)