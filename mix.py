def mix (vector<csv_reader>& csv, const vector<int>& vocabsize, ostream& out, timestep):
  #compute wordid offsets={0,V1,V1+V2,V1+V2+V3,....}
  vector<int>offset(vocabsize.size()+1,0);
  partial_sum(vocabsize.begin(), vocabsize.end(), offset.begin()+1);

  vector<int > next_time(csv.size());
  int last_time=0;
  int n = csv.size();
  bool done=true;
  for(int i=0;i<n;++i){
    csv[i].get();
    assert(!csv[i].words.empty());
    next_time[i]=csv[i].words[0];
  }
  size_t i = min_element(next_time.begin(), next_time.end()) - next_time.begin();
  int current_time = next_time[i];
  last_time = current_time;
  out<<current_time;

  bool first=true;
  while(!csv.empty()){
    size_t i = min_element(next_time.begin(), next_time.end()) - next_time.begin();
    current_time = next_time[i];

    if(timestep > 0)
      while(current_time > last_time + timestep){
        last_time += timestep;
        out<<endl<<last_time;
      }
    else if(current_time > last_time){
      last_time = current_time;
      out<<endl<<current_time;
    }

    for(size_t j=1;j<csv[i].words.size(); ++j){
      out<<","<<csv[i].words[j]+offset[i];
    }
    csv[i].get();

    if(csv[i].words.empty()){
      csv.erase(csv.begin()+i);
      next_time.erase(next_time.begin()+i);
    }
    else{
      next_time[i] = csv[i].words[0];
    }
  }
  out<<endl;

