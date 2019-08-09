import matplotlib.pyplot as plt
import numpy

# error graph
distances = numpy.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
errors = numpy.array([0, 0, 0.000390625, 0.000390625, 0.0015625, 0.001953125, 0.002734375, 0.005078125, 0.007421875, 0.009277344, 0.01796875, 0.028645833, 0.041015625])
error_exp = eval('(0.5 * 0.00000085) / (0.5 * 0.1 * 10 ** (-distances * 0.02) * 0.045 + 0.00000085)')

# throughput graph
throughputs = numpy.array([4159.540362,2904.709117,1804.995574,1120.451627,713.1432965,440.8206739,282.5234221,180.5635713,115.8317203,74.03500717,47.7472782,32.02199084,18.60194303])
throughputs_privacy = numpy.array([2191.282133,1449.536648,921.1564965,580.4188123,363.4383276,229.7218088,145.6966309,91.4210829,57.75562067,36.62583323,22.93101726,14.53338076,9.168929359])
latency = numpy.array([0.060176667,0.087206667,0.136013333,0.24242,0.334746667,0.551233333,0.93924,1.323406667,2.622773333,3.69686,5.853826667,7.881513333,12.8647])
latency_privacy = numpy.array([0.060249333,0.087686667,0.142453333,0.26242,0.35984,0.605533333,1.03248,1.48506,2.8832,4.14482,6.541826667,8.977333333,14.44462])

# measurement graph
width = 0.4
measured_0 = numpy.array([0.99, 0.026, 0.51, 0.51])
st_dev_0 = numpy.array([0.007378648, 0.018378732, 0.044969125, 0.044969125])
measured_0_color = numpy.array(['w', 'k', 'w', 'w'])
measured_plus = numpy.array([0.503, 0.503, 0.925, 0.063])
st_dev_plus = numpy.array([0.011547005, 0.011547005, 0.035355339, 0.037859389])
measured_plus_color = numpy.array(['w', 'w', 'w', 'k'])

# fidelity graph
fidelity = numpy.array([0.997, 0.970, 0.960, 0.960])
threshold = numpy.array([0.66] * 4)
labels = ["$|e\\rangle$", "$|l\\rangle$", "$|+\\rangle$", "$|-\\rangle$"]

if __name__ == "__main__":

    # Error
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.add_subplot()
    ln1 = ax.plot(distances, 100 * error_exp, 'k-', label="Expected Error Rate")
    ln2 = ax.plot(distances, 100 * error_exp + 3.3, 'k--', label="Expected Error Rate + Phase error")
    ln3 = ax.plot(distances, 100 * errors, 'bs', label="Simulated Error Rate")
    ln4 = ax.plot(distances, 100 * errors + 3.3, 'rD', label = "Simulated Error + Phase Error")

    plt.rcParams.update({'font.size': 12})
    lns = ln3 + ln1 + ln2 + ln4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    ax.set_xlabel("Fiber Length (km)")
    ax.set_ylabel("Qubit Error Rate (%)")
    ax.set_xlim(0, 120)
    ax.grid(color='0.75', linestyle='-', linewidth=1)

    fig.tight_layout()
    plt.savefig('plotting/QKD_error.pdf')

    # Throughput
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.add_subplot()

    ln1 = ax.plot(distances, throughputs, 'bs', label="Raw Bit Rate")
    ln2 = ax.plot(distances, throughputs_privacy, 'rD', label="Privacy Bit Rate")

    plt.rcParams.update({'font.size': 12})
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=8)

    ax.set_xlabel("Fiber Length (km)")
    ax.set_ylabel("Bit Rate (bit/s)")
    ax.set_yscale('log')
    ax.set_xlim(0, 120)
    ax.set_ylim(1, 10e3)
    ax.grid(color='0.75', linestyle='-', linewidth=1)

    fig.tight_layout()
    plt.savefig('plotting/QKD_throughput.pdf')

    # Latency
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.add_subplot()

    ln1 = ax.plot(distances, latency, 'bs', label="Latency")
    ln2 = ax.plot(distances, latency_privacy, 'rD', label="Privacy Latency")

    plt.rcParams.update({'font.size': 12})
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.set_xlabel("Fiber Length (km)")
    ax.set_ylabel("Latency (s)")
    # ax.set_yscale('log')
    ax.set_ylim(0, 15)
    ax.grid(color='0.75', linestyle='-', linewidth=1)

    fig.tight_layout()
    plt.savefig('plotting/QKD_latency.pdf')

    # measurement
    x_pos = [i for i, _ in enumerate(measured_0)]

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.bar(x_pos, 100 * measured_0, yerr=(st_dev_0 * 100), align='center', ecolor='k', capsize=10, width=width, color='skyblue')
    ax.grid(axis='y', color='0.75', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    for i, v in enumerate(measured_0):
        if 0 < (v + st_dev_0[i]) * 100 < 20:
            ax.text(i - 0.04, (v + st_dev_0[i]) * 100 + 5, str(100 * v) + " $\\pm$ " + str(round(100 * st_dev_0[i], 2)),
                    color='k', fontweight='bold', rotation='vertical')
        else:
            ax.text(i - 0.04, 5, str(100 * v) + " $\\pm$ " + str(round(100 * st_dev_0[i], 2)),
                    color='w', fontweight='bold', rotation='vertical')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% measured $|e\\rangle$")
    ax.set_ylim(0, 100)

    plt.savefig('plotting/Teleportation_measurement_z.pdf')

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.bar(x_pos, 100 * measured_plus, yerr=(st_dev_plus * 100), align='center', ecolor='k', capsize=10, width=width, color='indianred')
    ax.grid(axis='y', color='0.75', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    for i, v in enumerate(measured_plus):
        if 0 < (v + st_dev_plus[i]) * 100 < 20:
            ax.text(i - 0.04, (v + st_dev_0[i]) * 100 + 5, str(100 * v) + " $\\pm$ " + str(round(100 * st_dev_plus[i], 2)),
                    color='k', fontweight='bold', rotation='vertical')
        else:
            ax.text(i - 0.04, 5, str(100 * v) + " $\\pm$ " + str(round(100 * st_dev_plus[i], 2)),
                    color='w', fontweight='bold', rotation='vertical')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% measured $|+\\rangle$")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    plt.savefig('plotting/Teleportation_measurement_x.pdf')

    # fidelity
    x_pos = [i for i, _ in enumerate(fidelity)]

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.bar(x_pos, fidelity, width=width, color='skyblue')
    ax.axhline(y=0.66, color='k', linestyle='--')
    ax.grid(axis='y', color='0.75', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    for i, v in enumerate(fidelity):
        ax.text(i - 0.02, 0.53, str(v), color='w', fontweight='bold', rotation='vertical')

    ax.set_ylabel("Fidelity")
    plt.xticks(x_pos, labels)
    ax.set_ylim(0.5, 1)

    fig.tight_layout()
    plt.savefig('plotting/Teleportation_fidelity.pdf')
