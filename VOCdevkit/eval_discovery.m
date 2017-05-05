function [ corloc ] = eval_discovery

addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

corloc = zeros(1, VOCopts.nclasses);
for i = 1:VOCopts.nclasses
    cls = VOCopts.classes{i};
    
    corloc(i) = VOCevaldis(VOCopts, 'comp4', cls, 'trainval');  % compute and display PR
    
end

end


function [ corloc ] = VOCevaldis(VOCopts, id, cls, testset)

cp = sprintf(VOCopts.annocachepath, testset);

if exist(cp,'file')
    fprintf('%s: pr: loading ground truth\n',cls);
    load(cp,'gtids','recs');
else
    [gtids,t]=textread(sprintf(VOCopts.imgsetpath,testset),'%s %d');
    for i=1:length(gtids)
        % display progress
        if toc>1
            fprintf('%s: pr: load: %d/%d\n',cls,i,length(gtids));
            drawnow;
            tic;
        end

        % read annotation
        recs(i)=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    end
    save(cp,'gtids','recs');
end

fprintf('%s: pr: evaluating discovery\n',cls);

nimg=0;
gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
for i=1:length(gtids)
    % extract objects of class
    clsinds=strmatch(cls,{recs(i).objects(:).class},'exact');
    gt(i).BB=cat(1,recs(i).objects(clsinds).bbox)';
    gt(i).diff=[recs(i).objects(clsinds).difficult];
    gt(i).det=false(length(clsinds),1);
    nimg=nimg+double(size(gt(i).BB, 2)>0);
end

% load results
[ids,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath,id,cls),'%s %f %f %f %f');
BB=[b1 b2 b3 b4]' + 1;

nd = length(ids);
tp=zeros(nd,1);
for d = 1:nd
    i=strmatch(ids{d},gtids,'exact');
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end
    
    bb=BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 && ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=VOCopts.minoverlap
        %if ~gt(i).diff(jmax)
            %if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
	           % false positive (multiple detection)
            %end
        %end
    end
end

corloc = sum(tp) / nimg;

fprintf('corloc for class %s is %f.\n', cls, corloc);

end