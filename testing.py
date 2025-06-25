import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

def main():
        filename1 = sys.argv[1]

        data = pd.read_json(filename1, lines=True)

        data.dropna(inplace=True)
        y = data['numVisitors'].to_numpy()
        x = data['avg_tmax_2016'].to_numpy()
        z = data['avg_tmax_2017'].to_numpy()
        a = data['avg_tmin_2016'].to_numpy()
        b = data['avg_tmin_2017'].to_numpy()
        u = data['avg_prcp_2016'].to_numpy()
        v = data['avg_prcp_2017'].to_numpy()
        x = x/10
        z = z/10
        a = a/10
        b = b/10
        u = u /100
        v = v /100

        ############# Calculating the Linear regression Values ################

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(z, y)
        slope3, intercept3, r_value3, p_value3, std_err3 = linregress(a, y)
        slope4, intercept4, r_value4, p_value4, std_err4 = linregress(b, y)
        slope5, intercept5, r_value5, p_value5, std_err5 = linregress(u, y)
        slope6, intercept6, r_value6, p_value6, std_err6 = linregress(v, y)

        ############### Calculating the lowess smoothing ####################
        smooth1 = lowess(y, x)
        smooth2 = lowess(y, z)
        smooth3 = lowess(y, a)
        smooth4 = lowess(y, b)
        smooth5 = lowess(y, u)
        smooth6 = lowess(y, v)

        ######################### First Graph plot for TMax data #######################################

        plt.figure(figsize=(15, 5))  # Define the size of the figure
        plt.subplot(1, 3, 1)
        plt.title('Average TMax distribution')
        plt.scatter(x, y, color = 'red', label='Data 2016')
        plt.scatter(z, y, color = 'blue', label='Data 2017')

        # Plot the regression line
        plt.plot(x, slope * x + intercept, color='green', label='Linear Regression 2016')
        plt.plot(z, slope2 * z + intercept2, color='orange', label='Linear Regression 2017')

        # Plot lowess line
        plt.plot(smooth1[:, 0], smooth1[:, 1], color='black', label='LOESS Smoothed 2016')
        plt.plot(smooth2[:, 0], smooth2[:, 1], color='pink', label='LOESS Smoothed 2017')

        # Add labels and legend
        plt.xlabel('Average Max Temperature')
        plt.ylabel('Number of Visitors')
        plt.legend()

        ######################### Second Graph plot for TMin data #######################################

        plt.subplot(1, 3, 2)
        plt.title('Average TMin distribution')
        plt.scatter(a, y, color = 'red', label='Data 2016')
        plt.scatter(b, y, color = 'blue', label='Data 2017')

        # Plot the regression line
        plt.plot(a, slope3 * a + intercept3, color='green', label='Linear Regression 2016')
        plt.plot(b, slope4 * b + intercept4, color='orange', label='Linear Regression 2017')
     
        # Plot lowess line
        plt.plot(smooth3[:, 0], smooth3[:, 1], color='black', label='LOESS Smoothed 2016')
        plt.plot(smooth4[:, 0], smooth4[:, 1], color='pink', label='LOESS Smoothed 2017')
   
        # Add labels and legend
        plt.xlabel('Average Min Temperature')
        plt.ylabel('Number of Visitors')
        plt.legend()

        ######################### Third Graph plot for Prcp data #######################################

        plt.subplot(1, 3, 3)
        plt.title('Average Precipitation distribution')
        plt.scatter(u, y, color = 'red', label='Data 2016')
        plt.scatter(v, y, color = 'blue', label='Data 2017')

        # Plot the regression line
        plt.plot(u, slope5 * u + intercept5, color='green', label='Linear Regression 2016')
        plt.plot(v, slope6 * v + intercept6, color='orange', label='Linear Regression 2017')

        # Plot lowess line
        plt.plot(smooth5[:, 0], smooth5[:, 1], color='black', label='LOESS Smoothed 2016')
        plt.plot(smooth6[:, 0], smooth6[:, 1], color='pink', label='LOESS Smoothed 2017')
   
        # Add labels and legend
        plt.xlabel('Average Precipitation')
        plt.ylabel('Number of Visitors')
        plt.legend()

        # Show the plot
        plt.savefig('output_plot.png')

        u_log = np.log(u)
        v_log = np.log(v)

        n1 = stats.normaltest(x).pvalue
        n2 = stats.normaltest(z).pvalue
        n3 = stats.normaltest(a).pvalue
        n4 = stats.normaltest(b).pvalue
        n5 = stats.normaltest(u_log).pvalue
        n6 = stats.normaltest(v_log).pvalue


        ############################### TTest and P_vales for the data ###########################################
        t_stat, p_val = stats.ttest_ind(x, y)
        t_stat2, p_val2 = stats.ttest_ind(z, y)
        t_stat3, p_val3 = stats.ttest_ind(a, y)
        t_stat4, p_val4 = stats.ttest_ind(b, y)
        t_stat5, p_val5 = stats.ttest_ind(u, y)
        t_stat6, p_val6 = stats.ttest_ind(v, y)


        ############################## saving the results in an output file ######################################
        with open('testing_output.txt', 'w') as file:
                file.write('1. Tests on avgTMAX values for year 2016 \n')
                file.write('    a. Linegress result values: \n')
                file.write('            slope: {}\n'.format(slope))
                file.write('            intercept: {}\n'.format(intercept))
                file.write('            r_value: {}\n'.format(r_value))
                file.write('            p_value: {}\n'.format(p_value))
                file.write('            std_err: {}\n'.format(std_err))
                file.write('    b. Normaltest result values: \n')
                file.write('            p_value: {}\n'.format(n1))
                file.write('    c. ttest result values: \n')
                file.write('            t_stat: {}\n'.format(t_stat))
                file.write('            p_value: {}\n\n\n'.format(p_val))
                file.write('2. Tests on avgTMAX values for year 2017 \n')
                file.write('    a. Linegress result values: \n')
                file.write('            slope: {}\n'.format(slope2))
                file.write('            intercept: {}\n'.format(intercept2))
                file.write('            r_value: {}\n'.format(r_value2))
                file.write('            p_value: {}\n'.format(p_value2))
                file.write('            std_err: {}\n'.format(std_err2))
                file.write('    b. Normaltest result values: \n')
                file.write('            p_value: {}\n'.format(n2))
                file.write('    c. ttest result values: \n')
                file.write('            t_stat: {}\n'.format(t_stat2))
                file.write('            p_value: {}\n\n\n'.format(p_val2))
                file.write('3. Tests on avgTMIN values for year 2016 \n')
                file.write('    a. Linegress result values: \n')
                file.write('            slope: {}\n'.format(slope3))
                file.write('            intercept: {}\n'.format(intercept3))
                file.write('            r_value: {}\n'.format(r_value3))
                file.write('            p_value: {}\n'.format(p_value3))
                file.write('            std_err: {}\n'.format(std_err3))
                file.write('    b. Normaltest result values: \n')
                file.write('            p_value: {}\n'.format(n3))
                file.write('    c. ttest result values: \n')
                file.write('            t_stat: {}\n'.format(t_stat3))
                file.write('            p_value: {}\n\n'.format(p_val3))
                file.write('4. Tests on avgTMIN values for year 2017 \n')
                file.write('    a. Linegress result values: \n')
                file.write('            slope: {}\n'.format(slope4))
                file.write('            intercept: {}\n'.format(intercept4))
                file.write('            r_value: {}\n'.format(r_value4))
                file.write('            p_value: {}\n'.format(p_value4))
                file.write('            std_err: {}\n'.format(std_err4))
                file.write('    b. Normaltest result values: \n')
                file.write('            p_value: {}\n'.format(n4))
                file.write('    c. ttest result values: \n')
                file.write('            t_stat: {}\n'.format(t_stat4))
                file.write('            p_value: {}\n\n'.format(p_val4))
                file.write('5. Tests on avgPRCP values for year 2016 \n')
                file.write('    a. Linegress result values: \n')
                file.write('            slope: {}\n'.format(slope5))
                file.write('            intercept: {}\n'.format(intercept5))
                file.write('            r_value: {}\n'.format(r_value5))
                file.write('            p_value: {}\n'.format(p_value5))
                file.write('            std_err: {}\n'.format(std_err5))
                file.write('    b. Normaltest result values: \n')
                file.write('            p_value: {}\n'.format(n5))
                file.write('    c. ttest result values: \n')
                file.write('            t_stat: {}\n'.format(t_stat5))
                file.write('            p_value: {}\n\n'.format(p_val5))
                file.write('6. Tests on avgTMIN values for year 2017 \n')
                file.write('    a. Linegress result values: \n')
                file.write('            slope: {}\n'.format(slope6))
                file.write('            intercept: {}\n'.format(intercept6))
                file.write('            r_value: {}\n'.format(r_value6))
                file.write('            p_value: {}\n'.format(p_value6))
                file.write('            std_err: {}\n'.format(std_err6))
                file.write('    b. Normaltest result values: \n')
                file.write('            p_value: {}\n'.format(n6))
                file.write('    c. ttest result values: \n')
                file.write('            t_stat: {}\n'.format(t_stat6))
                file.write('            p_value: {}\n\n'.format(p_val6))

if __name__ == '__main__':
    main()
