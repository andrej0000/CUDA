/*********************************************************************
 *                             Zadanie 1                             *
 *********************************************************************/
/* 1. Napisac kernel 'niepoprawnyHistogram', ktory bedzie dokladna 
 * (ale rownolegla) kopia histogramu ze slajdow. Nie nalezy korzystac
 * z funkcji atomowych (inkrementacje licznikow nalezy zrobic
 * ''standardowo'' w pamieci global).
 * Po napisaniu uruchomic i przekonac sie ze NIE dziala.
 *
 * 2. Sprawdzic o ile wyniki oczekiwane (poprawna wersja na cpu) roznia 
 * sie od otrzymanych w wyniku wykonania kernela 'niepoprawnyHistogram'.
 * W szczegolnosci mozna sprawdzic jak ta roznica (co do rzedu wielkosci)
 * zalezy od liczby koszykow M.
 *
 * 3. Sprawdzic czas wykonania dla roznych M (4, 8, 128, 512, 2048)
 * oraz ustalonego N=(1<<20).
 */

/*********************************************************************
 *                             Zadanie 2                             *
 *********************************************************************/
/* 1. Napisac kernel 'prostyHistogram', ktory bedzie dokladna
 * (ale rownolegla) POPRAWNA kopia histogramu ze slajdow. Inkrementacje
 * liczby zliczen w poszczegolnych koszykach nalezy napisac z 
 * wykorzystaniem funkcji atomicAdd w pamieci glownej karty.
 * Po napisaniu uruchomic i przekonac sie ze tym razem dziala.
 *
 * 2. Sprawdzic czas wykonania dla roznych M ( 4, 8, 128, 512, 2048)
 * oraz ustalonego N. Porównać czasy z analogicznym eksperymentem
 * wykonanym dla kernela 'niepoprawnyHistogram' i zastanowic sie czy 
 * otrzymane wyniki maja sens.
 */

 /*********************************************************************
 *                             Zadanie 3                             *
 *********************************************************************/
/* 1. Napisac kernel 'lepszyHistogram'. W tej wersji każdy blok 
 * powinien stworzyc sobie wlasna lokalna, tablice koszykow w pamieci 
 * shared. Obliczenie histogramu (tzn. inkrementacja liczby w 
 * odpowiednich koszykach) powinno byc wykonane w dwoch etapach:
 * a) najpierw bloki licza 'lokalne' histogramy z wlasnego kawalka tablicy
 * danych z pamieci global (inkrementujac liczniki w shared funkcja atomicAdd).
 * b) nastepnie kazdy blok dodaje swoje lokalne zliczenia do ''glownego''
 * histogramu w pamieci global (znowu atomicAdd).
 *
 * Wskazowka: pamiec shared nie jest zerowana przy inicjalizacji.
 *
 * 2. Analogicznie jak w zad 1 i 2 sprawdzic czasy wykonania dla roznych
 * wartosci M. Czy dziala lepiej niz 'prostyHistogram'? Dlaczego?
 */
 
 /*********************************************************************
 *                              Ogolne                               *
 *********************************************************************/
/* 1. Przy sprawdzaniu wydajnosci mozna ustalic seed generatora liczb 
 * losowych.
 * 2. Ponizej znajduja sie parametry wywolania kerneli, najlepiej tak
 * napisac funkcje z zadan {1,2,3} zeby liczbaWatkowNaBlok oraz 
 * liczbaBlokow pozostala stala.
 * 3. Czas wykonania sprawdzamy komenda nvprof
 * 4. Wersja z zadania 3 nie zadziala na kartach z labu, nie obsluguja 
 * one funkcji atomowych w pamieci shared.
 */

/*********************************************************************
 *                             Dodatkowe                             *
 *********************************************************************/
 /* 1. Zastanowic sie czy dla ustalonego z gory N i M mozna przewidziec
 * jakie powinny byc optymalne parametry uruchomienia kernela z zad. 3?
 *
 * 2. Sprawdzic czy dla niewielkiej liczby koszykow (powiedzmy M=16) nie 
 * oplaca sie zmodyfikowac funkcji 'lepszyHistogram' dodajac wiecej 
 * (2? 4? 8?) lokalnych kopii histogramu (w pamieci shared)? Dlaczego?
 */

 /*********************************************************************
 *                    Jeszcze bardziej dodatkowe                     *
 *********************************************************************/
/*
 * 1. Napisac ogolna wersje histogramu, dzialajaca nie tylko dla floatow
 * z zakresu [0,1). Przed obliczeniem histogramu nalezy znac minimum i
 * maksimum dla tablicy z danymi, mozna je obliczyc na karcie wykonujac
 * redukcje wzgledem funkcji min oraz max.
 *
 * 2. Dodac mozliwosc zdefiniowania ''wlasnych'' koszykow, niekoniecznie 
 * stalej szerokosci.
 *
 * 3. Poprawic wydajnosc funkcji z zadan {1,2,3}, sprawdzic na przyklad:
 *    a) rozwiniecie petli (#pragma unroll),
 *    b) zmiane rozmiarow parametrow startowych kerneli
 * 
 * 4. Mozna na przyklad sprawdzic na ile optymalne sa domyslne parametry
 * wywolania (128,128), pomocny moze byc arkusz kalkulacyjny 
 * (google: cuda occupancy calculator).
 * W arkuszu podajemy architekture karty (na gromie jest to 2.0)
 * liczbe blokow, liczbe watkow oraz liczbe rejstrow jakie wykorzystuje
 * dany kernel (podaje sie zawsze liczbe rejestrow na 1 watek).
 * Informacje ile rejestrow wykorzystuje dany kernel mozna uzyskac 
 * kompilujac z flaga:
 * nvcc -Xptxas="-v"
 */

#include <omp.h>


/*********************************************************************
 *                    Parametry wywolania kerneli                    *
 *********************************************************************/
	const int liczbaWatkowNaBlok = 128;
	const int liczbaBlokow = 128;
	const int liczbaStrumieni = 1;

/*********************************************************************
 *                    Rozmiar danych i histogramu                    *
 *********************************************************************/
	int N=(1<<25); // Wielkosc danych do histogramowania
	int M = 1;    // Liczba koszyków


#include<cstdio>
#include<cstdlib>
#include<ctime>

__global__ void niepoprawnyHistogram(float * data, int * histo, int N, int M) {
	/* TODO: Tutaj nalezy wstawic napisany kod kernela z zadania 1 */
	/* data - wskaznik na dane do histogramowania (dlugosci N), nie modyfikowac
	 * histo - wskaznik do pamieci na histogram (dlugosci M) */
	int jump = gridDim.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < N; i += jump) {
		int koszyk = (int)(data[i] * M);
		histo[koszyk]++;
	}
}

__global__ void prostyHistogram(float * data, int * histo, int N, int M) {
	/* TODO: Tutaj nalezy wstawic napisany kod kernela z zadania 2 */
	/* data - wskaznik na dane do histogramowania (dlugosci N), nie modyfikowac
	 * histo - wskaznik do pamieci na histogram (dlugosci M) */
	int jump = gridDim.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < N; i += jump) {
		int koszyk = (int)(data[i] * M);
		atomicAdd(histo+koszyk, 1);
	}
	

}

__global__ void lepszyHistogram(float * data, int * histo, int N, int M) {
	/* TODO: Tutaj nalezy wstawic napisany kod kernela z zadania 3 */
	/* data - wskaznik na dane do histogramowania (dlugosci N), nie modyfikowac
	 * histo - wskaznik do pamieci na histogram (dlugosci M) */
	int size = N / gridDim.x;
	int start = size * blockIdx.x;
	int end = start + size;
	
	extern __shared__ int histoShared[];

	// Zerowanie shared
	for (int i = threadIdx.x; i < M; i += blockDim.x){
		histoShared[i] = 0;
	}

	__syncthreads();
	//Przeliczanie lokalnego
	//#pragma unroll
	for (int i = start + threadIdx.x; i < end; i += blockDim.x){
		int koszyk = (int)(data[i] * M);
		atomicAdd(histoShared + koszyk, 1);
	}
	
	//Czekamy az wszyscy skoncza
	__syncthreads();

	// Dodajemy nasz sharedHisto do globalnego

	for (int i = threadIdx.x; i < M; i += blockDim.x){
		int add = histoShared[i];
		atomicAdd(histo + i, add);
	}
	
}


//Sprawdzanie wynikow
void checkResults(int * cpuHisto, int * histo, int M) {
	for(int i=0;i<M;i++)
		if (cpuHisto[i] != histo[i] )
			printf("Blad! W komorce %d wyszlo %d, a powinno %d\n", i, histo[i], cpuHisto[i]);
}

int main() {
	srand(time(NULL));

	float * devData  ; //Wskaznik na dane do histogramowania na karcie
	float * data     ; //Wskaznik na dane do histogramowania na cpu
	int   * devHisto ; //Histogram na karcie
	int   * histo    ; //Miejsce do skopiowania histogramu z karty
	int   * cpuHisto ; //Referencyjny histogram na cpu (NIE MODYFIKOWAC)

	cudaMalloc(&devData   , N*sizeof(float));
	cudaMalloc(&devHisto  , M*sizeof(int));
	cudaMallocHost(&histo , M*sizeof(int));
	cudaMallocHost(&data  , N*sizeof(float));
	cpuHisto = (int*)malloc(M*sizeof(int));

	memset     (cpuHisto , 0 , M*sizeof(int)); //Zerujemy pamiec
	cudaMemset (devHisto , 0 , M*sizeof(int)); //Zerujemy pamiec na GPU

	//Generowanie floatow z zakresu [0,1)
	for(int i=0;i<N;i++) {
		data[i] = rand()/(float)(RAND_MAX);
	}
	//Kopiowanie wygenerowanych danych na karte (NIE MODYFIKOWAC) 
	//kolejne kernele korzystaja z tych samych danych, a to kopiowanie jest
	//tylko raz
	//Liczenie referencyjnego histogramu na CPU
	for(int i=0;i<N;i++) {
		cpuHisto[ (int)(data[i]*M) ]++;
	}
//
///*********************************************************************
// *                 Wywolanie 'niepoprawnyHistogram'                  *
// *********************************************************************/
//	cudaMemset (devHisto , 0 , M*sizeof(int)); //Zerujemy pamiec na GPU
//	
//	//Przykladowe wywolanie, zastanowic sie czy nie trzeba zmienic?
//	niepoprawnyHistogram<<<liczbaBlokow,liczbaWatkowNaBlok>>>( devData, devHisto, N, M);
//
//	//Kopiowanie gotowego histogramu na cpu
//	cudaMemcpy(histo, devHisto, M*sizeof(int), cudaMemcpyDeviceToHost);
//
//	//Sprawdzamy wyniki
//	printf("Sprawdzanie wynikow wygenerowanych przez 'niepoprawnyHistogram'\n");
//	checkResults(cpuHisto, histo, M);
//
//
///*********************************************************************
// *                    Wywolanie 'prostyHistogram'                    *
// *********************************************************************/
//	//Zerowanie histogramu na GPU
//	cudaMemset (devHisto , 0 , M*sizeof(int));
//	//Przykladowe wywolanie, zastanowic sie czy nie trzeba zmienic?
//	prostyHistogram<<<liczbaBlokow,liczbaWatkowNaBlok>>>( devData, devHisto, N, M);
//	//Kopiowanie gotowego histogramu na cpu
//	cudaMemcpy(histo, devHisto, M*sizeof(int), cudaMemcpyDeviceToHost);
//
//	printf("Sprawdzanie wynikow wygenerowanych przez 'prostyHistogram'\n");
//	checkResults(cpuHisto, histo, M);
//
/*********************************************************************
 *                    Wywolanie 'lepszyHistogram'                    *
 *********************************************************************/
	double timestart = omp_get_wtime();
	cudaStream_t strumienie[liczbaStrumieni];
	for(int i = 0; i < liczbaStrumieni; i++) {
		cudaStreamCreate(&(strumienie[i]));
	}
	//Zerowanie histogramu na GPU
	cudaMemset (devHisto , 0 , M*sizeof(int));
	for(int i = 0; i < liczbaStrumieni; i++) {
		int size = N/liczbaStrumieni;

		cudaMemcpyAsync(devData+(i*size), data+(i*size), size*sizeof(int), cudaMemcpyHostToDevice, strumienie[i]);
	
		//Przykladowe wywolanie, zastanowic sie czy nie trzeba zmienic?
		lepszyHistogram<<<liczbaBlokow,liczbaWatkowNaBlok, M*sizeof(int), strumienie[i]>>>( devData+(i*size), devHisto, size, M);
	}	
	//Kopiowanie gotowego histogramu na cpu
	cudaMemcpy(histo, devHisto, M*sizeof(int), cudaMemcpyDeviceToHost);
	
	double timeend = omp_get_wtime();
	
	printf("Czas %f\n", timeend - timestart);
	printf("Sprawdzanie wynikow wygenerowanych przez 'lepszyHistogram'\n");
	checkResults(cpuHisto, histo, M);
	

	for (int i = 0; i < liczbaStrumieni; i++) {
		cudaStreamDestroy(strumienie[i]);
	}

	//Zwalnianie pamieci
	cudaFreeHost(data)  ; 
	cudaFreeHost(histo) ; 
	cudaFree(devData)   ; 
	cudaFree(devHisto)  ; 
	free(cpuHisto)      ; 

	return 0;
}
