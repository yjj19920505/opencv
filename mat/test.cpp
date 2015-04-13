#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <iostream>
#include <stdio.h>//FILE
using namespace std;
using namespace cv;

#define featureDim 30*30
#define postiveNumber 11
#define negativeNumber 3
#define DEBUG 1
void LBP(Mat &image, float *result)  
{  
	int index = 0;
	for(int y = 1; y < image.rows-1; y ++)  
	{  
		for(int x = 1; x < image.cols-1; x++)  
		{  
			uchar neighbor[8] = {0};  
			neighbor[0] = image.at<uchar>(y-1, x-1);  
			neighbor[1] = image.at<uchar>(y-1, x);  
			neighbor[2] = image.at<uchar>(y-1, x+1);  
			neighbor[3] = image.at<uchar>(y, x+1);  
			neighbor[4] = image.at<uchar>(y+1, x+1);  
			neighbor[5] = image.at<uchar>(y+1, x);  
			neighbor[6] = image.at<uchar>(y+1, x-1);  
			neighbor[7] = image.at<uchar>(y, x-1);  
			uchar center = image.at<uchar>(y, x);  
			uchar temp = 0;  
			for(int k = 0; k < 8; k++)  
			{  
				temp += (neighbor[k] >= center)* (1<<k);  // ����LBP��ֵ  
			}  
			result[index++] = (float)temp;   
		}  
	}  
}  

void readTrainData(char * trainFilePath, float **&trainData, float *&trainLabels)
{
	int i, j;
	
	/*if (trainData != NULL)
	{
		delete []trainData;
	}*/
	trainData = new float*[postiveNumber + negativeNumber];
	trainLabels = new float[postiveNumber + negativeNumber];
	for (i = 0; i < (postiveNumber + negativeNumber); i++)
		trainData[i] = new float[featureDim];
	for (i = 0; i < postiveNumber; i++)
	{
		char fileName[100];
		//�������±���󣬿�ʼд����i,�����ļ������Ǵ�1��ʼ��
		sprintf(fileName, "%s\\pos\\img (%d).jpg", trainFilePath, i+1);
		
		if (DEBUG)
			cout << fileName << endl;
		IplImage *img = cvLoadImage(fileName);
		CvSize size;
		size.width = 30;
		size.height = 30;
		if (DEBUG && img)
			cout << img->depth <<endl;
		IplImage *resizedImage = cvCreateImage(size, img->depth, img->nChannels);
		cvResize(img, resizedImage, CV_INTER_LINEAR);
		for (int m = 0; m < size.height; m++)
		{
			for (int n = 0; n < size.width; n++)
			{
				CvScalar s = cvGet2D(resizedImage, m, n);//��ȡ���ص�Ϊ��n, m�����BGR��ֵ 
				
				trainData[i][m*size.width + n] = (float)(s.val[0] + s.val[1] + s.val[2])/3;
				
			}
		}
		trainLabels[i] = (float)1.0;
	}
	for (i = 0; i < negativeNumber; i++)
	{
		char fileName[100];
		sprintf(fileName, "%s\\neg\\img (%d).jpg", trainFilePath, i+1);
		
		if (DEBUG)
			cout << fileName << endl;
		IplImage *img = cvLoadImage(fileName);
		CvSize size;
		size.width = 30;
		size.height = 30;
		if (DEBUG && img)
			cout << img->depth <<endl;
		IplImage *resizedImage = cvCreateImage(size, img->depth, img->nChannels);
		cvResize( img, resizedImage, CV_INTER_LINEAR);
		
		
		for (int m = 0; m < size.height; m++)
		{
			for (int n = 0; n < size.width; n++)
			{
				CvScalar s = cvGet2D(resizedImage, m, n);//��ȡ���ص�Ϊ��n, m�����BGR��ֵ 
				
				trainData[i + postiveNumber][m*size.width + n] = (float)(s.val[0] + s.val[1] + s.val[2])/3;
				
			}
		}
		trainLabels[i + postiveNumber] = (float)0.0;
	}
	
}
//CvMat
void colorFilter(CvMat *inputImage, CvMat *&outputImage)
{
	int i, j;
	IplImage* image = cvCreateImage(cvGetSize(inputImage), 8, 3);
	cvGetImage(inputImage, image);    
    IplImage* hsv = cvCreateImage( cvGetSize(image), 8, 3 );  
	
    cvCvtColor(image,hsv,CV_BGR2HSV);
	int width = hsv->width;
	int height = hsv->height;
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			CvScalar s_hsv = cvGet2D(hsv, i, j);//��ȡ���ص�Ϊ��j, i�����HSV��ֵ 
			/*
				opencv ��H��Χ��0~180����ɫ��H��Χ�����(0~8)��(160,180) 
				S�Ǳ��Ͷȣ�һ���Ǵ���һ��ֵ,S���;��ǻ�ɫ���ο�ֵS>80)��
				V�����ȣ����;��Ǻ�ɫ�����߾��ǰ�ɫ(�ο�ֵ220>V>50)��|| (s_hsv.val[0]>160)&&(s_hsv.val[0]<180)) 
			*/
			CvScalar s;
			if (!(((s_hsv.val[0]>0)&&(s_hsv.val[0]<8)) || (s_hsv.val[0]>120)&&(s_hsv.val[0]<180)))
			{
				s.val[0] =0;
				s.val[1]=0;
				s.val[2]=0;
				cvSet2D(hsv, i ,j, s);
			}
			
			
		}
	outputImage = cvCreateMat( hsv->height, hsv->width, CV_8UC3 );
	cvConvert(hsv, outputImage);
	cvNamedWindow("filter");
	cvShowImage("filter", hsv);
	waitKey(0);
	cvReleaseImage(&hsv);
}
void colorFilter_Ipl(IplImage* image, IplImage*& hsv)
{
	int i, j;
	  
    hsv = cvCreateImage( cvGetSize(image), 8, 3 );  
	
    cvCvtColor(image,hsv,CV_BGR2HSV);
	int width = hsv->width;
	int height = hsv->height;
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			CvScalar s_hsv = cvGet2D(hsv, i, j);//��ȡ���ص�Ϊ��j, i�����HSV��ֵ 
			/*
				opencv ��H��Χ��0~180����ɫ��H��Χ�����(0~8)��(160,180) 
				S�Ǳ��Ͷȣ�һ���Ǵ���һ��ֵ,S���;��ǻ�ɫ���ο�ֵS>80)��
				V�����ȣ����;��Ǻ�ɫ�����߾��ǰ�ɫ(�ο�ֵ220>V>50)��|| (s_hsv.val[0]>160)&&(s_hsv.val[0]<180)) 
			*/
			CvScalar s;
			if (!(((s_hsv.val[0]>0)&&(s_hsv.val[0]<8)) || (s_hsv.val[0]>120)&&(s_hsv.val[0]<180)))
			{
				s.val[0] =0;
				s.val[1]=0;
				s.val[2]=0;
				cvSet2D(hsv, i ,j, s);
			}
			
			
		}
	
	cvNamedWindow("filter");
	cvShowImage("filter", hsv);
	waitKey(0);
	//cvReleaseImage(&hsv);
}
void test()
{
	
	IplImage *src= cvLoadImage("E:\\picture\\test_2.jpg");
	///*IplImage *src = NULL;
	//colorFilter_Ipl(image, src);*/
	if (src)
	{
		IplImage *dst = cvCreateImage(cvGetSize(src),8,1);
		IplImage *color_dst = cvCreateImage(cvGetSize(src),8,3);
		CvMemStorage *storage = cvCreateMemStorage();
		CvSeq *lines = 0;
		int i ;
		cvCanny(src,dst,50,200,3);
	
		cvCvtColor(dst,color_dst,CV_GRAY2BGR);
	#if 1
		lines = cvHoughLines2(dst,storage,CV_HOUGH_STANDARD,1,CV_PI/180,150,0,0);
		
		for (i=0;i<lines->total;i++)
		{
			float *line = (float *)cvGetSeqElem(lines,i);
			float rho = line[0];
			float theta = line[1];
			CvPoint pt1,pt2;
			double a = cos(theta);
			double b = sin(theta);
			if (fabs(a)<0.001)
			{
				pt1.x = pt2.x = cvRound(rho);
				pt1.y = 0;
				pt2.y = color_dst->height;
			}
			else if (fabs(b)<0.001)
			{
				pt1.y = pt2.y = cvRound(rho);
				pt1.x = 0;
				pt2.x = color_dst->width;
			}
			else
			{
				pt1.x = 0;
				pt1.y = cvRound(rho/b);
				pt2.x = cvRound(rho/a);
				pt2.y = 0;
			}

			cvLine(color_dst,pt1,pt2,CV_RGB(255,0,0),1,8);
		}
	#else
		lines = cvHoughLines2(dst,storage,CV_HOUGH_PROBABILISTIC,1,CV_PI/180,80,30,5);
		for (i=0;i<lines->total;i++)
		{
			CvPoint *line = (CvPoint *)cvGetSeqElem(lines,i);
			cvLine(color_dst,line[0],line[1],CV_RGB(255,0,0),1,CV_AA);
		}
	#endif
		cvNamedWindow("Source");
		cvShowImage("Source",src);

		cvNamedWindow("Hough");
		cvShowImage("Hough",color_dst);

		cvWaitKey(0);

		cvReleaseImage(&src);
		cvReleaseImage(&dst);
		cvReleaseImage(&color_dst);
		cvReleaseMemStorage(&storage);
		
		cvDestroyAllWindows();
		
		
	}

}
int main()
{
	//test();
	int i, j;
	float **trainData = NULL;
	float *trainLabels = NULL;
	readTrainData("E:\\picture", trainData, trainLabels);
	CvMat *trainDataMat = cvCreateMat(postiveNumber + negativeNumber, featureDim, CV_32FC1); 
	CvMat *labelsMat = cvCreateMat(postiveNumber + negativeNumber, 1, CV_32FC1); 
	 
	for (i = 0; i < (postiveNumber + negativeNumber); i++)
	{
		for (j = 0; j < featureDim; j++)
		{
			trainDataMat->data.fl[i*trainDataMat->cols + j] = trainData[i][j];
			/*if (DEBUG)
			cout <<  labelsMat->data.fl[i*labelsMat->cols + j]  <<":"<< trainData[i][j];*/
		}
		labelsMat->data.fl[i*labelsMat->cols] = (float)trainLabels[i];
		cout << labelsMat->data.fl[i] << ":" << trainLabels[i] << endl;
	}
	
    CvKNearest knn( trainDataMat, labelsMat, 0, false, 3 );

	 
	IplImage *testImage = cvLoadImage("E:\\picture\\test_2.jpg");
	CvMat *inputImage = cvCreateMat( testImage->height, testImage->width, CV_8UC3 );
	cvConvert(testImage, inputImage);
	CvMat *outputImage = NULL;
	colorFilter(inputImage, outputImage);
	


	if (!testImage)
	{
		cerr << "open image failed\n";
		return 0;
	}
	int height = testImage->height, width = testImage->width;
	int step_height = height / 4, step_width = width / 4;
	for (i = 0; i < (width - step_width/2); i += step_width)
	{	for (j = 0; j < (height - step_height/2); j += step_height)
		{
			//�ַ���С�ĵĴ����ˣ�һ��ʼд����height-i,����size.height = windowWidth;size.width = windowHeight;Ȼ��Ͳ��ϱ��ڴ�����Ժ�д���������Լ��Ķ�һ��,���ⲻҪ��㸴�ƴ���
			if (j > 300)
				cout << i << ":" << j << endl;
			int start_width = i, windowWidth = ((i+step_width)>width)?(width-i):step_width;
			int start_height = j, windowHeight = ((j+step_height)>height)?(height-j):step_height;
			if (DEBUG && (windowWidth > step_width || windowHeight > step_height))
			{
				printf("kkk start point:x%d,y%d. size:width%d, height%d\n", start_width, start_height, windowWidth, windowHeight);
			}
			CvSize size;
			size.height = windowHeight;
			size.width = windowWidth;
			cvSetImageROI(testImage,cvRect(start_width, start_height, windowWidth, windowHeight));//����Դͼ��ROI
			IplImage* pDest = cvCreateImage(size, testImage->depth, testImage->nChannels);//����Ŀ��ͼ��
			cvCopy(testImage, pDest); //����ͼ��
			cvResetImageROI(testImage);//Դͼ����������ROI
			
			
			CvSize sizeWindow;
			sizeWindow.height = 30;
			sizeWindow.width = 30;
			IplImage *resizedImage = cvCreateImage(sizeWindow, pDest->depth, pDest->nChannels);
			cvResize(pDest, resizedImage, CV_INTER_LINEAR);
			
			
			CvMat *testDataMat = cvCreateMat(1, featureDim, CV_32FC1);; 
			for (int m = 0; m < sizeWindow.height; m++)
			{
				for (int n = 0; n < sizeWindow.width; n++)
				{
					CvScalar s = cvGet2D(resizedImage, m, n);//��ȡ���ص�Ϊ��n, m�����BGR��ֵ 
				
					testDataMat->data.fl[m*sizeWindow.width + n] = (float)(s.val[0] + s.val[1] + s.val[2])/3;
				
				}
			}
		
			
			float result = knn.find_nearest(testDataMat, 1);
			if (DEBUG)
			cout << result << endl;
			if (result == 1)
			{
				
				CvPoint startPoint, pointSize;
				startPoint.x = start_width;
				startPoint.y = start_height;
				pointSize.x = startPoint.x + windowWidth;
				pointSize.y = startPoint.y + windowHeight;
				if (DEBUG)
					printf("start point:x%d,y%d. size:width%d, height%d\n", startPoint.x, startPoint.y, pointSize.x, pointSize.y);
				//cvCircle( testImage, startPoint, 30, CV_RGB(255,255,0));
				cvRectangle(testImage, startPoint, pointSize, CV_RGB(0,255,0));   
			}
		}
	}
	cvNamedWindow("test", 1);
	cvShowImage("test", testImage);
	waitKey(0);
	system("pause");
	return 0;
}

