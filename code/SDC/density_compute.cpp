#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<algorithm>
#include<vector>
#include<chrono>
#include<sstream>
using namespace std;
//#define N 3861 // the number of dataset
//#define D 2 // the dimension
//string rate = "0.1";
//string data_name = "overlap2";
struct Point
{
	double dis;
	int idx;
	double dsum;
	int idxSortedByDim;
};

struct AnnularannHead {
	int head;
	int length;
};

double computeDistance(double a[], double b[], int D) {
	double sumation = 0.0;
	for (int i = 0; i < D; i++) {
		sumation += pow(a[i] - b[i], 2);
	}
	return sqrt(sumation);
}

bool compare_points_dis(const Point& p1, const Point& p2) {
	return p1.dis < p2.dis;
}

bool compare_points_dim(const Point& p1, const Point& p2) {
	return p1.dsum < p2.dsum;
}

void computeDensity(double** data, double point[], double r, int* dense, int N, int D) {
	Point* points = new Point[N];
	for (int i = 0; i < N; i++) {
		points[i].dis = -1;
		points[i].dsum = -1;
		points[i].idx = -1;
		points[i].idxSortedByDim = -1;
	}
	/*      calculate the distance between objects and virtual object
			calculate the sumation of dimesional values
			sort objects by the distance to virtual object
	*/
	for (int i = 0; i < N; i++) {
		points[i].idx = i;
		points[i].dis = computeDistance(data[i], point, D);
		double sumation = 0;
		for (int j = 0; j < D; j++) {
			sumation += data[i][j];
		}
		points[i].dsum = sumation;
	}
	sort(points, N + points, compare_points_dis);
	
	/*  get the number (tt) of public annular regions */
	double max_dis = 0.;
	double min_dis = 1e9;
	for (int i = 0; i < N; i++) {
		max_dis = max_dis < points[i].dis ? points[i].dis : max_dis;
		min_dis = min_dis > points[i].dis ? points[i].dis : min_dis;
	}
	int tt = ceil((max_dis - min_dis) / r);

	/*   paritioning public annular regions,record the index of the start object of each annular region
		 and the length of each annular region */
	AnnularannHead* annHead = new AnnularannHead[tt];
	for (int i = 0; i < tt; i++) {
		annHead[i].length = 0;
		annHead[i].head = 0;
	}
	int flag = 0;
	int length = 0;
	double annular_gap = r;
	annHead[0].head = 0;
	for (int i = 0; i < N; i++) {
		while (points[i].dis - min_dis > annular_gap) {
			annular_gap += r;
			annHead[flag+1].head = i;
			annHead[flag++].length = length;
			length = 0;
		}
		length++;
	}
	annHead[flag].length = length;
	if (annHead[tt - 1].length == 0) {
		tt--;
	}
	cout << "total annular " << tt << endl;

	/*     density calculation     */
	for (int i = 0; i < tt; i++) {
		int head = 0;
 		int tail = 0;
		if (annHead[i].length == 0) {
			continue;
		}
		if (i == 0) {
			head = 0;
			tail = annHead[1].head + annHead[1].length;
		}
		else if (i == tt - 1) {
			tail = N;
			head = annHead[tt - 2].head;
		}
		else if (i == tt - 2) {
			tail = N;
			head = annHead[tt - 3].head;
		}
		else {
			head = annHead[i].head - annHead[i - 1].length;
			tail = annHead[i + 1].head + annHead[i + 1].length;
		}
		/*  sort the three public annular regions according to sumation of dimensional values */
		int* indeces = new int[tail - head];
		for (int j = 0; j < tail - head; j++) {
			indeces[j] = j;
		}
		sort(indeces, indeces + (tail - head), [&points, head](int a, int b) {
			return points[a + head].dsum < points[b + head].dsum;
			}
		);
		for (int j = 0; j < tail - head; j++) {// ��������ӳ���points�ж�Ӧ�ĵ�
			points[head + indeces[j]].idxSortedByDim = j;//����points�еĵ�i����֪�����ڰ�ά�Ⱥ�������λ��
		}
		
		/*    calculate the density of all objects in the i-th public annular region  */
		int end = tail - head;
		int start = annHead[i].head;// the start position in points of current three public annular regions 
		for (int j = 0; j < annHead[i].length; j++) {
			int curIdx = start + j;// the position of current object in points
			int pre = points[curIdx].idxSortedByDim - 1;// the position in indeces of the previous object (pre) of current object
			int lat = points[curIdx].idxSortedByDim + 1;//  the position in indeces of the latter object (lat) of current object
			
			int preIdx = 0;// the position of pre in points
			int latIdx = 0;// the position of lat in points
			while (pre >= 0 && abs(points[head + indeces[pre]].dsum - points[curIdx].dsum) <= sqrt(D) * r) {
				preIdx = head + indeces[pre];
				if (computeDistance(data[points[preIdx].idx], data[points[curIdx].idx], D) <= r) {
					dense[points[curIdx].idx] += 1;
				}
				pre--;
			}
			while (lat < end && abs(points[head + indeces[lat]].dsum - points[curIdx].dsum) <= sqrt(D) * r) {
				latIdx = head + indeces[lat];
				if (computeDistance(data[points[latIdx].idx], data[points[curIdx].idx], D) <= r) {
					dense[points[curIdx].idx] += 1;
				}
				lat++;
			}
		}
		delete[] indeces;
	}
	delete[] annHead;
	delete[] points;
}

int main(int argc, char* argv[]) {

	ifstream infile;
	//string file_path = (string)argv[1] + (string)argv[2] + "/miss_" + (string)argv[3] + "_complete_objects.txt";
	//string save_path = (string)argv[1] + (string)argv[2] + "/dense_" + (string)argv[3] + ".txt";
	string file_path = (string)argv[1] + (string)argv[2] + ".txt";
	string save_path = (string)argv[1] + (string)argv[2] + "_dense.txt";
	int N = stoi(argv[4]);
	int D = stoi(argv[5]);
	double r = stof(argv[6]);
	double ratio = stof(argv[3]);
	cout << "data path: " << file_path << endl<< "save path: " << save_path <<endl 
		<< "N: " << N << "  D: " << D << endl << "r: " << r << endl << "ratio: " << ratio << endl;
	double* point = new double[D];
	double** data = new double* [N];
	for (int i = 0; i < N; i++) {
		data[i] = new double[D];
		for (int j = 0; j < D; j++) {
			data[i][j] = 0;
		}
	}
	infile.open(file_path, ios::in);
	if (!infile) {
		cout << "open file '"<<file_path<<"' error!" << endl;
		return 0;
	}
	string buf;
	int row = 0;
	while (getline(infile, buf)) {
		double num;
		istringstream iss(buf);
		int col = 0;
		while (iss >> num && col < D) {
			data[row][col] = num;
			col++;
		}
		row++;
		/*for (int i = 0; i <= buf.length(); i++) {
			if (buf[i] == '\0') {
				data[count++][dim] = stof(item);
				item = "";
				dim = 0;
			}
			else if (buf[i] == ' ') {
				data[count][dim++] = stof(item);
				item = "";
			}
			else {
				item = item + buf[i];
			}
		}*/
	}
	cout << "row:" << row << endl;
	cout << "col:" << D << endl;
	infile.close();
	double min_s = 1e5;
	for (int i = 0; i < D; i++) {
		for (int j = 0; j < N; j++) {
			if (min_s > data[j][i]) {
				min_s = data[j][i];
			}
			point[i] = min_s;
		}
	}
	int* dense = new int[N];
	int* dense_start = dense;
	for (int i = 0; i < N; i++) {
		dense[i] = 0;
	}
	auto start_time = chrono::high_resolution_clock::now();
	computeDensity(data, point, r, dense, N, D);
	auto end_time = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
	cout << "the runtime of density calculation program: " << duration.count() <<"ms"<< endl;
	
	ofstream fout(save_path);
	if (!fout) {
		cout << "open file '" << save_path << "' error!" << endl;
		return 0;
	}
	for (int i = 0; i < N; i++) {
		fout << *dense << endl;
		dense++;
	}
	fout << duration.count() << endl;
	fout.close();
	for (int i = 0; i < N; i++) {
		delete[] data[i];
	}
	delete[] point;
	delete[] data;
	delete[] dense_start;
	return 0;
}