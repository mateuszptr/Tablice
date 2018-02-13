
#include "Tablice.h"

using namespace cv;


/** @function main */
int main(int argc, char** argv) {
	/// Load an image
	Mat src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat color = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!src.data) {
		return -1;
	}
	Mat clahe;
	set_clahe(src, clahe);

	//Mat unsharp;
	//set_unsharp(clahe, unsharp);

	imshow("src", src);
	imshow("clahe", clahe);
	//imshow("unsharp", unsharp);

	Mat otsu, otsu2;
	//threshold(unsharp, otsu, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	//threshold(unsharp, otsu2, 192, 255, CV_THRESH_BINARY_INV);
	adaptiveThreshold(clahe, otsu2, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C,
			CV_THRESH_BINARY_INV, 27, 0);
//
//	Mat blur2;
//	GaussianBlur(otsu, blur2, Size(3.0,3.0), 1.0);
//	threshold(blur2, otsu2, 0,255,CV_THRESH_OTSU);
//
//	imshow("otsu", otsu);
	imshow("threshhold", otsu2);

	Mat labelImage, stats, centroids;
	connectedComponentsWithStats(otsu2, labelImage, stats, centroids, 8);

	std::vector<CCA> ccas;
	create_cca(labelImage, stats, centroids, ccas);
	filter_cca(ccas, labelImage);

	display_blobs(labelImage, ccas, "CCA");

	std::vector<PlateCandidate> pcs;
	find_plate_candidates(ccas, pcs);
	filter_candidates(ccas, pcs);

	display_blobs(labelImage, ccas, "Groups");
	display_image_with_rects(color, pcs);

	Mat ocr_mat(labelImage.size(), CV_8UC3);
	get_matrix(labelImage, ccas, ocr_mat);
	//imshow("ss", ocr_mat);
	ocr(ocr_mat, pcs);

	waitKey(0);
	return 0;
}
