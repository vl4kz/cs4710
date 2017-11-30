def main():
    k_results = []
    for i in range(6):
        with open("kfolds_" + str(i) + ".txt", 'r') as f:
            k_results.append(float(f.readline().strip()))
    print(sum(k_results)/len(k_results))

if __name__ == '__main__':
    main()
