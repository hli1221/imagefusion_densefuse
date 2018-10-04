function MI = analysis_MI(A,B,F)

% MI_A = nmi(A,F);
% MI_B = nmi(B,F);

% MI_A = mutual_information_images(A,F);
% MI_B = mutual_information_images(B,F);

MI_A = MutualInformation(A,F);
MI_B = MutualInformation(B,F);

MI = MI_A + MI_B;

end

% by soleimani h.soleimani@ec.iut.ac.ir
%input---> im1 and im2... they should be in gray scale,[0 255], and have the same size
function MI=mutual_information_images(im1, im2)
im1=double(im1)+1;
im2=double(im2)+1;

% find joint histogram
joint_histogram=zeros(256,256);

for i=1:min(size(im1,1),size(im2,1))
    for j=1:min(size(im1,2),size(im2,2))
       joint_histogram(im1(i,j),im2(i,j))= joint_histogram(im1(i,j),im2(i,j))+1;
    end
end


 JPDF=joint_histogram/sum(joint_histogram(:)); % joint pdf of two images
 pdf_im1=sum(JPDF,1); % pdf of im1
 pdf_im2=sum(JPDF,2); % pdf of im2
 
 % find MI
 MI=0;
 for i=1:256
     for j=1:256
         if JPDF(i,j)>0
             MI=MI+JPDF(i,j)*log2(JPDF(i,j)/(pdf_im1(i)*pdf_im2(j)));
         end
     end
 end
end

% MutualInformation: returns mutual information (in bits) of the 'X' and 'Y'
% by Will Dwinnell
%
% I = MutualInformation(X,Y);
%
% I  = calculated mutual information (in bits)
% X  = variable(s) to be analyzed (column vector)
% Y  = variable to be analyzed (column vector)
%
% Note: Multiple variables may be handled jointly as columns in matrix 'X'.
% Note: Requires the 'Entropy' and 'JointEntropy' functions.
%
% Last modified: Nov-12-2006

function I = MutualInformation(X,Y)

if (size(X,2) > 1)  % More than one predictor?
    % Axiom of information theory
    I = JointEntropy(X) + entropy(Y) - JointEntropy([X Y]);
else
    % Axiom of information theory
    I = entropy(X) + entropy(Y) - JointEntropy([X Y]);
end


% God bless Claude Shannon.

% EOF
end


% JointEntropy: Returns joint entropy (in bits) of each column of 'X'
% by Will Dwinnell
%
% H = JointEntropy(X)
%
% H = calculated joint entropy (in bits)
% X = data to be analyzed
%
% Last modified: Aug-29-2006

function H = JointEntropy(X)

% Sort to get identical records together
X = sortrows(X);

% Find elemental differences from predecessors
DeltaRow = (X(2:end,:) ~= X(1:end-1,:));

% Summarize by record
Delta = [1; any(DeltaRow')'];

% Generate vector symbol indices
VectorX = cumsum(Delta);

% Calculate entropy the usual way on the vector symbols
H = entropy(VectorX);


% God bless Claude Shannon.

% EOF
end