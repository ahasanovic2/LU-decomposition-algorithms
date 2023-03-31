#include <iostream>
#include <chrono>
#include <memory>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <random>

using std::cout;
using std::endl;
using std::cin;

constexpr auto eps = 0.0001;

/*---------------------------------------------------HELPING-------------------------------------------------------------------*/

class Timer {
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1, 1000000>> second_;
    std::chrono::time_point<clock_> beg_;
    const char* header;
public:
    Timer(const char* header = "") : beg_(clock_::now()), header(header) {}
    ~Timer() {
        double e = elapsed();
        //cout << header << ": " << e / 1000000 << " seconds" << endl;
    }
    void reset() {
        beg_ = clock_::now();
    }
    double elapsed() const {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }
};

double** create_initial_matrix(int dim) {
    double** matrica = new double* [dim];
    for (int i = 0; i < dim; i++) {
        matrica[i] = new double[dim];
    }
    return matrica;
}

void print_matrix(double** pointer, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) {
            cout << pointer[i][j] << ", ";
        }
        cout << pointer[i][n - 1] << endl;
    }
}

void generate_random_matrix(double** matrica, int dim) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(1, 1000);


    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            matrica[i][j] = distribution(generator);
        }
    }
}

void make_symmetric_matrix(double** matrica, int dim) {
    for (int i(0); i < dim; i++)
        for (int j(0); j < dim; j++)
            if(i != j)
                matrica[i][j] = matrica[j][i];
}

void deallocate_memory(double** matrica, int dim) {
    for (int i = 0; i < dim; i++) {
        delete[] matrica[i];
    }
    delete[] matrica;
}

void compute_product(double** L, double** U, double** A, int n) {
    int i, j, k;
    double sum = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sum = 0;
            for (k = 0; k < n; k++) {
                sum = sum + L[i][k] * U[k][j];
            }
            A[i][j] = sum;
        }
    }
}

double** copy_matrix(double** matrica, int dim) {
    double** kopija = create_initial_matrix(dim);
    for (int i(0); i < dim; i++)
        for (int j(0); j < dim; j++)
            kopija[i][j] = matrica[i][j];
    return kopija;
}

void save_matrix_to_file(double** A, int n, std::string file_name) {
    std::ofstream output(file_name, std::ios::out);
    if (output.fail()) {
        std::cerr << "Failed to open the file" << endl;
    }
    else {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - 1; j++) {
                output << A[i][j] << ",";
            }
            output << A[i][n - 1] << endl;
        }
        // Close the file stream
        output.close();
    }

}

bool is_matrix_symmetric(double** matrica, int n) {
    for (int i(0); i < n; i++) {
        for (int j(0); j < n; j++) {
            if (i != j)
                if (fabs(matrica[i][j] - matrica[j][i]) > eps)
                    return false;
        }
    }
    return true;
}

/*---------------------------------------------------CHOLESKY------------------------------------------------------------------*/

bool cholesky1_sequential(double** matrica, double** L_seq, int n) {
    bool provjera(true);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;

            if (j == i) // summation for diagonals
            {
                for (int k = 0; k < j; k++)
                    sum += pow(L_seq[j][k], 2);
                L_seq[j][j] = sqrt(matrica[j][j] - sum);
            }
            else {
                // Evaluating L(i, j) using L(j, j)
                for (int k = 0; k < j; k++)
                    sum += (L_seq[i][k] * L_seq[j][k]);
                L_seq[i][j] = (matrica[i][j] - sum) / L_seq[j][j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        if (L_seq[i][i] <= 0) {
            provjera = false;
            break;
        }
    }
    return provjera;
}

bool cholesky1_parallel(double** matrica, double** L_par, int n) {
    bool provjera(true);
    double sum(0);
    int i, j, k;
#pragma omp parallel for shared(matrica,L_par) private(i,j,k,sum) schedule(dynamic)
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            sum = 0;

            if (j == i) // summation for diagonals
            {
                for (k = 0; k < j; k++)
                    sum += pow(L_par[j][k], 2);
                L_par[j][j] = sqrt(matrica[j][j] - sum);
            }
            else {
                // Evaluating L(i, j) using L(j, j)
                for (k = 0; k < j; k++)
                    sum += (L_par[i][k] * L_par[j][k]);
                L_par[i][j] = (matrica[i][j] - sum) / L_par[j][j];
            }
        }
    }
    for (i = 0; i < n; i++) {
        if (L_par[i][i] <= 0) {
            provjera = false;
            break;
        }
    }
    return provjera;
}

void test_cholesky_time(const int &dim) {
    auto matrica = create_initial_matrix(dim);
    generate_random_matrix(matrica, dim);
    make_symmetric_matrix(matrica, dim);
    double sequential_time(0);
    
    {
        auto L_seq = create_initial_matrix(dim);
        {
            Timer t("Cholesky sequential time");
            cholesky1_sequential(matrica, L_seq, dim);
            sequential_time = t.elapsed();
        }
        deallocate_memory(L_seq, dim);
    }

    std::vector<double> computation_time;
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 2; j <= 8; j *= 2) {
                auto L_par = create_initial_matrix(dim);
                omp_set_dynamic(0);
                omp_set_num_threads(j);
                {
                    Timer t("CHOLESKY 1 PARALLEL");
                    cholesky1_parallel(matrica, L_par, dim);
                    computation_time.emplace_back(t.elapsed());
                }
                deallocate_memory(L_par, dim);
            }
        }
    }
    cout << "Cholesky sequential time" << ": " << sequential_time / 1000000 << " seconds" << endl;
    double time_2th(0), time_4th(0), time_8th(0);
    for (int i = 0; i < computation_time.size(); i+=3) {
        time_2th += computation_time.at(i);
        time_4th += computation_time.at(i+1);
        time_8th += computation_time.at(i+2);
    }
    cout << "Cholesky parallel time (2 threads)" << ": " << time_2th / ((computation_time.size() / 3) * 1000000) << " seconds" << endl;
    cout << "Cholesky parallel time (4 threads)" << ": " << time_4th / ((computation_time.size() / 3) * 1000000) << " seconds" << endl;
    cout << "Cholesky parallel time (8 threads)" << ": " << time_8th / ((computation_time.size() / 3) * 1000000) << " seconds" << endl;

    cout << endl << "Speedup is: " << endl;
    cout << "2 threads: " << (sequential_time / 1000000)/(time_2th / ((computation_time.size() / 3) * 1000000)) << endl;
    cout << "4 threads: " << (sequential_time / 1000000)/(time_4th / ((computation_time.size() / 3) * 1000000)) << endl;
    cout << "8 threads: " << (sequential_time / 1000000)/(time_8th / ((computation_time.size() / 3) * 1000000)) << endl;

    deallocate_memory(matrica, dim);
}

void test_cholesky_accuracy() {
    for (;;) {
        cout << endl << "Insert dimension of square matrix (0 for exit): " << endl;
        int dim;
        std::cin >> dim;
        if (!std::cin || dim < 0) {
            cout << "Invalid input. Try again" << endl;
            cin.clear();
            cin.ignore();
            continue;
        }
        else if (dim == 0)
            break;
        auto matrica = create_initial_matrix(dim);
        generate_random_matrix(matrica, dim);
        make_symmetric_matrix(matrica, dim);
        auto L_seq = create_initial_matrix(dim);
        auto L_par = create_initial_matrix(dim);
        if (cholesky1_sequential(matrica, L_seq, dim)) {
            cholesky1_parallel(matrica, L_par, dim);
            bool provjera(true);
            for (int i(0); i < dim; i++) {
                for (int j(0); j <= i; j++) {
                    if (fabs(L_seq[i][j] - L_par[i][j]) > 1e-4) {
                        provjera = false;
                        i = dim, j = dim;
                    }
                }
            }
            if (provjera)
                cout << endl << "Success! Matrices are identical." << endl;
            else
                cout << endl << "ERROR! Matrices are not identical!" << endl;
        }
        else
            cout << endl << "Cannot compute Cholesky on given matrix" << endl;

        deallocate_memory(matrica, dim);
        deallocate_memory(L_seq, dim);
        deallocate_memory(L_par, dim);
    }
}

/*-----------------------------------------------------CROUT-------------------------------------------------------------------*/

int crout_sequential(double** A, double** L, double** U, int n) {
    int i, j, k;
    double sum = 0;
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }
    for (j = 0; j < n; j++) {
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[i][k] * U[k][j];
            }
            L[i][j] = A[i][j] - sum;
        }
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[j][k] * U[k][i];
            }
            if (L[j][j] == 0) {
                return 0;
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
    return 1;
}

int crout_parallel(double** A, double** L, double** U, int n) {
    int i, j, k;
    double sum = 0;
    bool provjera(true);
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }
    for (j = 0; j < n; j++) {
#pragma omp parallel for private(i, k, sum) shared(A, L, U, j) schedule(static)
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[i][k] * U[k][j];
            }
            L[i][j] = A[i][j] - sum;
        }
#pragma omp parallel for private(i, k, sum) shared(A, L, U, j) schedule(static)
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[j][k] * U[k][i];
            }
            if (L[j][j] == 0) {
                provjera = false;
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
    return provjera;
}

void test_crout_time(const int &dim) {
    auto matrica = create_initial_matrix(dim);
    generate_random_matrix(matrica, dim);
    make_symmetric_matrix(matrica, dim);
    double sequential_time(0);
    {
        auto L_seq = create_initial_matrix(dim);
        auto U_seq = create_initial_matrix(dim);
        {
            Timer t("Crout sequential time");
            crout_sequential(matrica, L_seq, U_seq, dim);
            sequential_time = t.elapsed();
        }
        deallocate_memory(L_seq, dim);
        deallocate_memory(U_seq, dim);
    }
    std::vector<double> computation_time;
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 2; j <= 8; j *= 2) {
                auto L_par = create_initial_matrix(dim);
                auto U_par = create_initial_matrix(dim);
                omp_set_dynamic(0);
                omp_set_num_threads(j);
                {
                    Timer t("Crout parallel time");
                    crout_parallel(matrica, L_par, U_par, dim);
                    computation_time.emplace_back(t.elapsed());
                }
                deallocate_memory(L_par, dim);
                deallocate_memory(U_par, dim);
            }
        }
    }

    deallocate_memory(matrica, dim);

    cout << "Crout sequential time" << ": " << sequential_time / 1000000 << " seconds" << endl;
    double time_2th(0), time_4th(0), time_8th(0);
    for (int i = 0; i < computation_time.size(); i += 3) {
        time_2th += computation_time.at(i);
        time_4th += computation_time.at(i + 1);
        time_8th += computation_time.at(i + 2);
    }
    cout << "Crout parallel time (2 threads)" << ": " << time_2th / ((computation_time.size() / 3) * 1000000) << " seconds" << endl;
    cout << "Crout parallel time (4 threads)" << ": " << time_4th / ((computation_time.size() / 3) * 1000000) << " seconds" << endl;
    cout << "Crout parallel time (8 threads)" << ": " << time_8th / ((computation_time.size() / 3) * 1000000) << " seconds" << endl;

    cout << endl << "Speedup is: " << endl;
    cout << "2 threads: " << (sequential_time / 1000000) / (time_2th / ((computation_time.size() / 3) * 1000000)) << endl;
    cout << "4 threads: " << (sequential_time / 1000000) / (time_4th / ((computation_time.size() / 3) * 1000000)) << endl;
    cout << "8 threads: " << (sequential_time / 1000000) / (time_8th / ((computation_time.size() / 3) * 1000000)) << endl;
}

void test_crout_accuracy() {
    for (;;) {
        cout << endl << "Insert dimension of square matrix (0 for exit): " << endl;
        int dim;
        std::cin >> dim;
        if (!std::cin || dim < 0) {
            cout << "Invalid input. Try again" << endl;
            cin.clear();
            cin.ignore();
            continue;
        }
        else if (dim == 0)
            break;
        auto matrica = create_initial_matrix(dim);
        generate_random_matrix(matrica, dim);
        make_symmetric_matrix(matrica, dim);
        auto L_seq = create_initial_matrix(dim);
        auto U_seq = create_initial_matrix(dim);
        auto L_par = create_initial_matrix(dim);
        auto U_par = create_initial_matrix(dim);
        crout_sequential(matrica, L_seq, U_seq, dim);
        crout_parallel(matrica, L_par, U_par, dim);
        bool provjera(true);
        auto A_seq = create_initial_matrix(dim);
        auto A_par = create_initial_matrix(dim);
        compute_product(L_seq, U_seq, A_seq, dim);
        compute_product(L_par, U_par, A_par, dim);
        for (int i(0); i < dim; i++) {
            for (int j(0); j <= i; j++) {
                if (fabs(L_seq[i][j] - L_par[i][j]) > 1e-2) {
                    provjera = false;
                    cout << endl << "i = " << i << ", j = " << j << endl;
                    cout << endl << "A_seq[i][j] = " << A_seq[i][j] << ", A_par[i][j] = " << A_par[i][j] << endl;
                    i = dim, j = dim;
                }
            }
        }

        if (provjera)
            cout << endl << "Success! Matrices are identical." << endl;
        else
            cout << endl << "ERROR! Matrices are not identical!" << endl;

        deallocate_memory(matrica, dim);
        deallocate_memory(L_seq, dim);
        deallocate_memory(U_seq, dim);
        deallocate_memory(A_seq, dim);
        deallocate_memory(L_par, dim);
        deallocate_memory(U_par, dim);
        deallocate_memory(A_par, dim);
    }
}

/*-----------------------------------------------------LDLT--------------------------------------------------------------------*/

bool ldlt_sequential(double** A, const int n) {
    bool provjera(true);
    double* w = new double[n] {0};
    for (int i = 0; i < n; i++) {
        double s;
        for (int j = 0; j <= i - 1; j++) {
            s = A[i][j];
            for (int k = 0; k <= j - 1; k++) {
                s -= w[k] * A[j][k];
            }
            w[j] = s;
            A[i][j] = (double)s / A[j][j];
        }
        s = A[i][i];
        for (int k = 0; k <= i - 1; k++) {
            s -= w[k] * A[i][k];
        }
        A[i][i] = s;
        if (A[i][i] <= 0) {
            provjera = false;
        }
    }
    delete[] w;
    return provjera;
}

void test_ldlt_time(int n) {
    double** A = create_initial_matrix(n);
    make_symmetric_matrix(A, n);
    double** LDLT_par = copy_matrix(A, n);
    double sequential_time(0);
    {
        Timer t("LDLT Sequential time");
        ldlt_sequential(A, n);
        sequential_time = t.elapsed();
    }
    cout << "Cholesky sequential time" << ": " << sequential_time / 1000000 << " seconds" << endl;

    deallocate_memory(A, n);
    deallocate_memory(LDLT_par, n);
}

/*--------------------------------------------------MAIN-FUNCTIONS-------------------------------------------------------------*/

void automatic_factorisation(double** matrica, int dim) {
    auto matrica_ldlt = copy_matrix(matrica, dim);
    auto matrica_cholesky = create_initial_matrix(dim);
    auto simetricnost = is_matrix_symmetric(matrica, dim);
    if (simetricnost && cholesky1_parallel(matrica, matrica_cholesky, dim)) {
        cout << "Cholesky factorization successful!" << endl;
        save_matrix_to_file(matrica_cholesky, dim, "output.txt");
    }
    else if (simetricnost && ldlt_sequential(matrica_ldlt, dim)) {
        cout << "Cholesky factorization not possible!" << endl;
        cout << "LDLT factorization successful!" << endl;
        save_matrix_to_file(matrica_ldlt, dim, "output.txt");
    }
    else {
        auto L = create_initial_matrix(dim);
        auto U = create_initial_matrix(dim);
        crout_parallel(matrica, L, U, dim);
        cout << "LDLT factorization unsuccessful" << endl;
        cout << "Cholesky factorization unsuccessful" << endl;
        cout << "Crout factorization successful!" << endl;
        save_matrix_to_file(L, dim, "output_L.txt");
        save_matrix_to_file(U, dim, "output_U.txt");
        deallocate_memory(L, dim);
        deallocate_memory(U, dim);
    }
    deallocate_memory(matrica_ldlt, dim);
    deallocate_memory(matrica_cholesky, dim);

}

void compute_factorisation(double** matrica, int dim) {
    for (;;) {
        bool terminirati(false);
        cout << "Choose factorisation you want to compute: " << endl;
        cout << "1. Crout factorisation" << endl;
        cout << "2. Cholesky factorisation" << endl;
        cout << "3. LDLT factorisation" << endl;
        cout << "4. Automatic" << endl;
        cout << "0. Exit" << endl;
        int opcija;
        cin >> opcija;
        switch (opcija) {
            case 1: {
                auto L = create_initial_matrix(dim);
                auto U = create_initial_matrix(dim);
                crout_parallel(matrica, L, U, dim);
                save_matrix_to_file(L, dim, "output_L.txt");
                save_matrix_to_file(U, dim, "output_U.txt");
                deallocate_memory(L, dim);
                deallocate_memory(U, dim);
                break;
            }
            case 2: {
                auto L = create_initial_matrix(dim);
                auto provjera = cholesky1_parallel(matrica, L, dim);
                if (provjera && is_matrix_symmetric(matrica,dim)) {
                    save_matrix_to_file(L, dim, "output.txt");
                }
                else
                    cout << endl << "Cholesky factorization not possible for provided matrix." << endl;
                deallocate_memory(L, dim);
                break;
            }
            case 3: {
                auto matrica_copy = copy_matrix(matrica, dim);
                if (is_matrix_symmetric(matrica_copy,dim) && ldlt_sequential(matrica_copy, dim)) {
                    auto matrica_L = create_initial_matrix(dim);
                    auto matrica_D = create_initial_matrix(dim);
                    for (int i(0); i < dim; i++) {
                        for (int j(0); j <= i; j++) {
                            if (j < i)
                                matrica_L[i][j] = matrica_copy[i][j];
                            else
                                matrica_D[i][j] = matrica_copy[i][j];
                        }
                    }
                    save_matrix_to_file(matrica_L, dim, "output_L.txt");
                    save_matrix_to_file(matrica_D, dim, "output_D.txt");
                    deallocate_memory(matrica_L, dim);
                    deallocate_memory(matrica_D, dim);
                }
                else {
                    cout << endl << "LDLT factorization not possible for provided matrix." << endl;
                }
                deallocate_memory(matrica_copy,dim);
                break;
            }
            case 4: {
                automatic_factorisation(matrica, dim);
                break;
            }
            case 0: {
                terminirati = true;
                break;
            }
            default: {
                cout << "Unknown option. Try again. " << endl;
                cin.clear();
                cin.ignore();
                break;
            }
        }
        if (terminirati)
            break;
    }
}

void solve_and_save_result() {
    for (;;) {
        bool terminirati(false);
        cout << "How do you want to insert matrix:" << endl;
        cout << "1. Insert manually" << endl;
        cout << "2. Read from a file" << endl;
        cout << "3. Random generate" << endl;
        cout << "0. Exit" << endl;
        int opcija;
        cin >> opcija;
        switch (opcija) {
            case 1: {
                int dim;
                for (;;) {
                    cout << "Insert dimension for matrix:";
                    cin >> dim;
                    if (!cin || dim <= 0) {
                        cin.clear();
                        cin.ignore();
                        cout << "Invalid input. Try again." << endl;
                        continue;
                    }
                    break;
                }
                auto matrica = create_initial_matrix(dim);
                cout << "Insert numbers: ";
                for (int i(0); i < dim; i++) {
                    for (int j(0); j < dim; j++) {
                        double unos_broja;
                        for (;;) {
                            cin >> unos_broja;
                            if (!cin) {
                                cin.clear();
                                cin.ignore();
                                cout << "Invalid input. Try again." << endl;
                                continue;
                            }
                            break;
                        }
                        matrica[i][j] = unos_broja;
                    }
                }
                compute_factorisation(matrica, dim);
                deallocate_memory(matrica, dim);
                break;
            }
            case 2: {
                int dim;
                for (;;) {
                    cout << "Insert dimension for matrix:";
                    cin >> dim;
                    if (!cin || dim <= 0) {
                        cin.clear();
                        cin.ignore();
                        cout << "Invalid input. Try again." << endl;
                        continue;
                    }
                    break;
                }
                auto A = create_initial_matrix(dim);
                std::ifstream input_file("matrix.txt");
                std::vector<std::vector<double>> matrix;
                std::string line;
                while (std::getline(input_file, line)) {
                    std::vector<double> row;
                    std::stringstream line_stream(line);
                    std::string cell;
                    while (std::getline(line_stream, cell, ',')) {
                        row.push_back(stoi(cell));
                    }
                    matrix.push_back(row);
                }
                input_file.close();
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        A[i][j] = matrix.at(i).at(j);
                    }
                }
                compute_factorisation(A, dim);
                deallocate_memory(A, dim);
            }
            case 3: {
                int rows;
                for (;;) {
                    cout << "Insert number of rows for matrix: ";
                    cin >> rows;
                    if (!cin || rows <= 0) {
                        cin.clear();
                        cin.ignore();
                        cout << "Invalid input. Try again." << endl;
                        continue;
                    }
                    break;
                }
                auto A = create_initial_matrix(rows);
                generate_random_matrix(A, rows);;
                compute_factorisation(A, rows);
                deallocate_memory(A, rows);
                break;
            }
            case 0: {
                terminirati = true;
                break;
            }
            default: {
                cout << "Invalid input. Try again." << endl;
                cin.clear();
                cin.ignore();
                break;
            }
        }
        if (terminirati)
            break;
    }
}

void calculate_computation_time() {
    for (;;) {
        cout << endl << "Insert dimension of square matrix (0 for exit): " << endl;
        int dim;
        std::cin >> dim;
        if (!std::cin || dim < 0) {
            cout << "Invalid input. Try again" << endl;
            cin.clear();
            cin.ignore();
            continue;
        }
        else if (dim == 0)
            break;
        cout << endl << "For which method you want to measure time:" << endl;
        cout << "1. Crout algorithm" << endl;
        cout << "2. Cholesky algorithm" << endl;
        cout << "3. LDLT algorithm" << endl;
        int opcija;
        std::cin >> opcija;
        switch (opcija) {
        case 1: {
            test_crout_time(dim);
            break;
        }
        case 2: {
            test_cholesky_time(dim);
            break;
        }
        case 3: {
            test_ldlt_time(dim);
            break;
        }
        default: {
            cout << "Invalid input. Try again" << endl;
            cin.clear();
            cin.ignore();
            break;
        }
        }
    }
}

void calculate_accuracy() {
    for (;;) {
        cout << endl << "For which method you want to calculate accuracy:" << endl;
        cout << "1. Crout algorithm" << endl;
        cout << "2. Cholesky algorithm" << endl;
        cout << "0. Exit" << endl;
        int opcija;
        std::cin >> opcija;
        switch (opcija) {
            case 1: {
                test_crout_accuracy();
                break;
            }
            case 2: {
                test_cholesky_accuracy();
                break;
            }
            case 0: {
                return;
            }
            default: {
                cout << "Invalid input. Try again" << endl;
                cin.clear();
                cin.ignore();
                break;
            }
        }
    }
}

/*------------------------------------------------------MAIN-------------------------------------------------------------------*/

int main()
{
    cout << "Welcome!" << endl;
    for (;;) {
        bool terminirati(false);
        cout << endl << "Insert option:" << endl;
        cout << "1. Calculate computation time of different methods of LU factorisation" << endl;
        cout << "2. Solve and save result" << endl;
        cout << "3. Check accuracy for algorithms" << endl;
        cout << "0. Exit" << endl;
        int opcija1;
        std::cin >> opcija1;
        switch (opcija1) {
            case 1: {
                calculate_computation_time();
                break;
            }
            case 2: {
                solve_and_save_result();
                break;
            }
            case 3: {
                calculate_accuracy();
                break;
            }
            case 0: {
                return 0;
            }
            default: {
                cout << "Invalid input. Try again." << endl;
                break;
            }
        }
    }
    return 0;
}