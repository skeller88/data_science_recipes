import pandas as pd


def statsmodels_output_table(output):
    output_table = pd.DataFrame(dict(coefficients=output.params,
                             SE=output.bse, z=output.tvalues,
                             p_values=output.pvalues))
    significant = output_table.loc[output_table['p_values'] >= 0.05].sort_values("coefficients", ascending=False)
    not_significant = output_table.loc[output_table['p_values'] < 0.05].sort_values("coefficients", ascending=False)
    return significant, not_significant
