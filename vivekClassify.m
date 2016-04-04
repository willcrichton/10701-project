%%%%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%
MAXWINSIZE = 40;

%%%%%%%%%%%%%%% Lambda function for calculating R2 %%%%%%%%%%%%%%%%
rsquare = @(y, f) max(0,1 - sum((y(:)-f(:)).^2)/sum((y(:)-mean(y(:))).^2));
rmse = @(y, f) sqrt((sum((y(:) - f(:)).^2) ) ./ size(y,1)-1);

%%%%%%%%%%%%%%% Prep Data for Analysis %%%%%%%%%%%%%%%%%

filename = 'Crisis_data_ACLED Version 6 All Africa 1997-2015_csv_dyadic.csv';
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);

%Keep columns: Date, Country, Location, Latitude, Longitude, Notes
dataArray = A.textdata(:, [4, 15, 19, 20, 21, 24]);

%Remove first row - header information
data = cell2table(dataArray(2:end, :));

%Seperate dates from day/month/year into 3 seperate values
dates = rowfun(@(x) strsplit(char(x), '/'), data(:, 1));
%Covert to array as under rowfun sets output as one variable, instead of 3
dataArray  = table2array(dates);
dataArray(:, [1,2]) = dataArray(:, [2,1]); %Switch month and day
dateTable = cell2table(dataArray);

%Concatenate the dates to the original table
data = [dateTable, data];


%%%%%%%%%%%%%%% Prep Data for Analysis %%%%%%%%%%%%%%%%%

%Choose all events after, including 2015 for dataset

years = ['2010'; '2011'; '2012'; '2013'; '2014'; '2015',]; %2016 not includeing in this dataset
mask2010 = strcmp(data.dataArray3, '2010');
mask2011 = strcmp(data.dataArray3, '2011');
mask2012 = strcmp(data.dataArray3, '2012');
mask2013 = strcmp(data.dataArray3, '2013');
mask2014 = strcmp(data.dataArray3, '2014');
mask2015 = strcmp(data.dataArray3, '2015');

mask = mask2010 + mask2011 + mask2012 +  mask2013 +  mask2014 + mask2015;
assert(max(mask) == 1)

dataMasked = data(logical(mask), :);

%The number of days since the start of 2014. Use this to count the number
%of events a day
dateCat = rowfun(@(x,y,z) days365(strcat('01-01-', years(1, :)), strcat(x,'-',y,'-',z)), dataMasked(:, 1:3));

%We see that there exists atleast one event every day
max(table2array(dateCat))
min(table2array(dateCat))
numel(unique((table2array(dateCat))))


%%%%%%%%%%%%%%% Prepare Feature, Result Matrices %%%%%%%%%%%%%%%%%
%                  Train Logistic Regression                     %

%Work with array instead of table
dataArray = table2array(dateCat);
%Count the number of crsis that occur on each data
[countElem, elem] = histcounts(dataArray,unique(dataArray));
countElem = countElem';


%Iterate over window size, cross validate to choose best parameter
%%%% SINGLE FEATURE REGRESSION - sum of number of events in last win size
%%%% days

r2Mat = zeros(MAXWINSIZE, 1);
rmseMat = zeros(MAXWINSIZE, 1);

for winsize = 1:MAXWINSIZE
       
    numObs = size(countElem, 1) - winsize ;
    X = zeros(numObs, 1);
    Y = zeros(numObs, 1);
    
    for slide = 1 : numObs;
    
            X(slide) = sum(countElem(slide : slide + winsize - 1));
            Y(slide) = countElem(slide + winsize);
      
    end
    
    %Split into train and test set. Standard 80, 20 split
    [trainInd,valInd,testInd] = dividerand(numObs, 0.8, 0, 0.2);

    XTrain = X(trainInd', :);
    YTrain = Y(trainInd', :);
    XTest = X(testInd', :);
    YTest = Y(testInd', :);

    % Get weights of regression
    weights = mldivide(XTrain, YTrain);
    predict = XTest * weights;
    r2Mat(winsize) = rsquare(YTest, predict);
    rmseMat(winsize) = rmse(YTest, predict);
   
end
display(max(r2Mat))
display(r2Mat)
scatter([1:MAXWINSIZE], r2Mat);

%%%% MULTI FEATURE REGRESSION - number of events in last win size
%%%% days

r2Mat = zeros(MAXWINSIZE, 1);
rmseMat = zeros(MAXWINSIZE, 1);

for winsize = 1:MAXWINSIZE
       
    numObs = size(countElem, 1) - winsize ;
    X = zeros(numObs, winsize);
    Y = zeros(numObs, 1);
    
    for slide = 1 : numObs;
    
            X(slide, :) = (countElem(slide : slide + winsize - 1))';
            Y(slide) = countElem(slide + winsize);
      
    end
    
    %Split into train and test set. Standard 80, 20 split
    [trainInd,valInd,testInd] = dividerand(numObs, 0.8, 0, 0.2);

    XTrain = X(trainInd', :);
    YTrain = Y(trainInd', :);
    XTest = X(testInd', :);
    YTest = Y(testInd', :);

    % Get weights of regression
    weights = mldivide(XTrain, YTrain);
    predict = XTest * weights;
    r2Mat(winsize) = rsquare(YTest, predict);
    rmseMat(winsize) = rmse(YTest, predict);
   
end
display(r2Mat)
scatter([1:MAXWINSIZE], r2Mat);
hold on;
scatter([1:MAXWINSIZE], rmseMat);
hold off;
