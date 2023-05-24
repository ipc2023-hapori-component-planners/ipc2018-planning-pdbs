#include "pattern_collection_generator_genetic_Online_SS.h"

#include "validation.h"
#include "zero_one_pdbs.h"

#include "../causal_graph.h"
#include "../globals.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"

#include "../utils/markup.h"
#include "../utils/math.h"
#include "../utils/rng.h"
#include "../utils/timer.h"
#include "../heuristic.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <math.h>
//Hack to use SS get_type, it needs heuristic object in constructor
#include "../heuristics/blind_search_heuristic.h"
#include "../heuristics/lm_cut_heuristic.h"
#include "../successor_generator.h"
#include "../utils/countdown_timer.h"
#include "pdb_factory.h"
#include "pattern_database_interface.h"
#include "../utils/debug_macros.h"

    
using namespace std;

namespace pdbs {
    PatternCollectionGeneratorGeneticSS::PatternCollectionGeneratorGeneticSS(
	const Options &opts)
	: pdb_factory (opts.get<shared_ptr<PDBFactory>>("pdb_factory")),
	  recompute_max_additive_subsets(opts.get<bool>("recompute_max_additive_subsets")), 
	  pdb_max_size(opts.get<double>("pdb_max_size")),
	  num_collections(opts.get<int>("num_collections")),
	  num_episodes(opts.get<int>("num_episodes")),
	  mutation_probability(opts.get<double>("mutation_probability")),
	  disjoint_patterns(opts.get<bool>("disjoint")), 
	  hybrid_pdb_size(opts.get<bool>("hybrid_pdb_size")),
	  time_limit(opts.get<int>("time_limit")),
	  genetic_time_limit(opts.get<int>("time_limit")){

	cout<<"Setting num_collections to 1 no matter the input,peak memory:"<<utils::get_peak_memory_in_kb()<<endl; // ???????
	cout<<"hybrid_pdb_size:"<<hybrid_pdb_size<<endl;
	num_collections=1;
	////result=make_shared<PatternCollectionInformation>(task, make_shared<PatternCollection>());
	if(pdb_factory->name()=="symbolic"){
	  pdb_max_size=2*pow(10,5);
	}
	if(recompute_max_additive_subsets)
	  cout<<"recompute_max_additive_subsets is on"<<endl;
	else
	  cout<<"recompute_max_additive_subsets is off"<<endl;

	cout<<"time limit for symbolic pdbs:"<<time_limit<<endl;

	genetic_SS_timer = new utils::CountdownTimer(genetic_time_limit);
    }

    void PatternCollectionGeneratorGeneticSS::select(
	const vector<double> &fitness_values) {
	if(num_collections==1){//No point selecting if  only one collection!
	    return;
	}
	vector<double> cumulative_fitness;
	cumulative_fitness.reserve(fitness_values.size());
	double total_so_far = 0;
	for (double fitness_value : fitness_values) {
	    total_so_far += fitness_value;
	    cumulative_fitness.push_back(total_so_far);
	}
	// total_so_far is now sum over all fitness values.

	vector<vector<vector<bool>>> new_pattern_collections;
	new_pattern_collections.reserve(num_collections);
	//cout<<"num_collections:"<<num_collections<<",fitness_values.size:"<<fitness_values.size()<<",pattern_collections.size:"<<pattern_collections.size()<<endl;fflush(stdout);
	for (int i = 0; i < num_collections; ++i) {
	    int selected;
	    if (total_so_far == 0) {
		// All fitness values are 0 => choose uniformly.
		selected = (*g_rng())(fitness_values.size());
	    } else {
		// [0..total_so_far)
		double random = (*g_rng())() * total_so_far;
		// Find first entry which is strictly greater than random.
		selected = upper_bound(cumulative_fitness.begin(),
				       cumulative_fitness.end(), random) -
		    cumulative_fitness.begin();
	    }
	    //cout<<"selected:"<<selected<<endl;fflush(stdout);
	    new_pattern_collections.push_back(pattern_collections[selected]);
	}
	pattern_collections.swap(new_pattern_collections);
    }

    void PatternCollectionGeneratorGeneticSS::mutate() {
	for (auto &collection : pattern_collections) {
	    for (vector<bool> &pattern : collection) {
		for (size_t k = 0; k < pattern.size(); ++k) {
		    double random = (*g_rng())(); // [0..1)
		    if (random < mutation_probability) {
			pattern[k].flip();
		    }
		}
	    }
	}
    }
    int PatternCollectionGeneratorGeneticSS::mutate2() {
	int mutations=0;
	for (size_t i = 0; i < pattern_collections.size(); ++i) {
	    for (size_t j = 0; j < pattern_collections[i].size(); ++j) {
		vector<bool> &pattern = pattern_collections[i][j];
		vector<bool> orig_pattern=pattern;
		for (size_t k = 0; k < pattern.size(); ++k) {
		    double random = (*g_rng())(); // [0..1)
		    if (random < mutation_probability) {
			pattern[k].flip();
		   
			//Check if new pattern has any irrelevant irrelevant variables
			vector<int> trans_pattern;
			trans_pattern=transform_to_pattern_normal_form(pattern);
			remove_irrelevant_variables(trans_pattern);
			transform_to_pattern_bitvector_form(pattern,trans_pattern);
			if(pattern!=orig_pattern){
			    mutations++;
			}
		    }
		}
	    }
	}
	return mutations;
    }

    // Alvaro: All these functions could be moved somewhere else
    // (pattern.h?) and be shared with genetic
    void PatternCollectionGeneratorGeneticSS::transform_to_pattern_bitvector_form(vector<bool> &bitvector,
										  const vector<int> &pattern) const {
	bitvector.assign(g_variable_name.size(), false);
	for (size_t i = 0; i < pattern.size(); ++i) {
	    bitvector[pattern[i]]=true;
	}
    }

    Pattern PatternCollectionGeneratorGeneticSS::transform_to_pattern_normal_form(
	const vector<bool> &bitvector) const {
	Pattern pattern;
	for (size_t i = 0; i < bitvector.size(); ++i) {
	    if (bitvector[i]){
		pattern.push_back(i);
	    }
	}
	return pattern;
    }

    void PatternCollectionGeneratorGeneticSS::remove_irrelevant_variables(
	Pattern &pattern) const {
	TaskProxy task_proxy(*task);

	unordered_set<int> in_original_pattern(pattern.begin(), pattern.end());
	unordered_set<int> in_pruned_pattern;

	vector<int> vars_to_check;
	for (FactProxy goal : task_proxy.get_goals()) {
	    int var_id = goal.get_variable().get_id();
	    if (in_original_pattern.count(var_id)) {
		// Goals are causally relevant.
		vars_to_check.push_back(var_id);
		in_pruned_pattern.insert(var_id);
	    }
	}

	while (!vars_to_check.empty()) {
	    int var = vars_to_check.back();
	    vars_to_check.pop_back();
	    /*
	      A variable is relevant to the pattern if it is a goal variable or if
	      there is a pre->eff arc from the variable to a relevant variable.
	      Note that there is no point in considering eff->eff arcs here.
	    */
	    const CausalGraph &cg = task_proxy.get_causal_graph();

	    const vector<int> &rel = cg.get_eff_to_pre(var);
	    for (size_t i = 0; i < rel.size(); ++i) {
		int var_no = rel[i];
		if (in_original_pattern.count(var_no) &&
		    !in_pruned_pattern.count(var_no)) {
		    // Parents of relevant variables are causally relevant.
		    vars_to_check.push_back(var_no);
		    in_pruned_pattern.insert(var_no);
		}
	    }
	}

	pattern.assign(in_pruned_pattern.begin(), in_pruned_pattern.end());
	sort(pattern.begin(), pattern.end());
    }

    bool PatternCollectionGeneratorGeneticSS::is_pattern_too_large(
	const Pattern &pattern) const {
	// Test if the pattern respects the memory limit.
	TaskProxy task_proxy(*task);
	VariablesProxy variables = task_proxy.get_variables();
	double mem = 1;
	for (size_t i = 0; i < pattern.size(); ++i) {
	    VariableProxy var = variables[pattern[i]];
	    double domain_size = var.get_domain_size();
	    if (!utils::is_product_within_limit(mem, domain_size, pdb_max_size))
		return true;
	    mem *= domain_size;
	}
	return false;
    }

    bool PatternCollectionGeneratorGeneticSS::mark_used_variables(
	const Pattern &pattern, vector<bool> &variables_used) const {
	for (size_t i = 0; i < pattern.size(); ++i) {
	    int var_id = pattern[i];
	    if (variables_used[var_id])
		return true;
	    variables_used[var_id] = true;
	}
	return false;
    }

    void PatternCollectionGeneratorGeneticSS::evaluate(vector<double> &fitness_values) {
	//static int last_count=best_heuristic.count_zero_one_pdbs();
	//static int last_sampled_best_heurs_count=0;
	//static double last_time_better_heur_found=0;
	int raised_states=0;
	int sampled_states=0;
	int current_heur_initial_value=-1;
	//cout<<"calling evaluate_genetic_online_SS"<<endl;
	double min_size=0;
	bool skip_sampling=false;
	static int valid_pattern_counter=0;
	//static bool print_timer=false;
      
	/*if(pdb_factory->name()!="symbolic"){
	  if(hybrid_pdb_size){
	    //cout<<"hybrid_pdb_size is set"<<endl;
	      if(utils::g_timer()<100.0){
		  //min_size=1;
		  pdb_max_size=50000; 
	      }
	      else if(utils::g_timer()<200){
		  //min_size=200000;
		  pdb_max_size=500000;
	      }
	      else if(utils::g_timer()<350){
		  //min_size=400000;
		  pdb_max_size=1000000;
	      }
	      else if(utils::g_timer()<500.0){
		  pdb_max_size=9*pow(10,7);
	      }
	      else{
		  pdb_max_size=9*pow(10,8);
	      } 
	  }

	  if(last_sampler_too_big){
	      pdb_max_size=last_pdb_max_size;
	      min_size=last_pdb_min_size;
	  }

	  if(valid_pattern_counter>5){
	    min_size=pdb_max_size/1000;
	  }
	  else{
	    min_size=0;
	  }
    }
    else{//so symbolic*/
      //pdb_max_size=numeric_limits<double>::max();
      if(!last_sampler_too_big){
	if(bin_packed_episode){
	  pdb_max_size*=10;
	  min_size=pdb_max_size/100;
	  cout<<"pdb_max_size raised to:"<<pdb_max_size<<endl;
	}
      }
      min_size=1;//avoid empty patterns
    //}
    
   DEBUG_MSG(cout<<"evaluate, pdb_max_size:"<<pdb_max_size<<",min_size:"<<min_size<<endl<<flush;);
  //pdb_max_size=double(INT_MAX)*double(100);

	//DEBUG_MSG(cout<<"setting pdb_max_size to:"<<pdb_max_size<<endl;);
	TaskProxy task_proxy(*task);
	int collection_counter=0;
	for (const auto &collection : pattern_collections) {
	    //DEBUG_MSG(cout << "evaluate pattern collection " << (collection_counter + 1) << " of " << pattern_collections.size() << endl;fflush(stdout););
	    DEBUG_MSG(cout << "evaluate pattern collection " << (collection_counter + 1) << " of " << pattern_collections.size() << endl;fflush(stdout););
	    double overall_pdb_size=0;
	    double fitness = 0;
	    bool pattern_valid = true;
	    vector<double> probe_avgs;
	    double pruned_states=0;
	    double total_SS_gen_nodes=0;
	    vector<bool> variables_used(task_proxy.get_variables().size(), false);
	    int best_heur_dead_ends=0;


	    DEBUG_MSG(cout<<"making pattern_collection"<<endl;);
	    shared_ptr<PatternCollection> pattern_collection = make_shared<PatternCollection>();
	    pattern_collection->reserve(collection.size());
	    for (const vector<bool> &bitvector : collection) {
		//cout<<"working on bitvector:"<<bitvector<<endl;
		Pattern pattern = transform_to_pattern_normal_form(bitvector);
		//cout<<"transformed Pattern:"<<pattern<<endl;

		//if(pdb_factory->name()!="symbolic"){
		  if (is_pattern_too_large(pattern)) {
		      DEBUG_MSG(cout << "pattern exceeds the memory limit!,pdb_max_size:" << pdb_max_size<<endl;);
		      pattern_valid = false;
		      break;
		  }
		//}
		
		remove_irrelevant_variables(pattern);

		if (disjoint_patterns) {
		    if (mark_used_variables(pattern, variables_used)) {
			//cout << "patterns are not disjoint anymore!" << endl;
			pattern_valid = false;
			break;
		    }
		}

		pattern_collection->push_back(pattern);
		overall_pdb_size+=get_pattern_size(pattern);
	    }
	    DEBUG_MSG(cout<<"finished making pattern_collection"<<endl;);
	    if (!pattern_valid) {
		/* Set fitness to a very small value to cover cases in which all
		   patterns are invalid. */
		pattern_valid=false;
		fitness = 0.001;
	    }
	    else if(overall_pdb_size<min_size){
		//cout<<"pattern collection size:"<<overall_pdb_size<<" too small, skipping"<<endl;
		fitness = 0.001;
	    } else {
	
		/*
		  std::pair<set< vector<Pattern> >,bool > ret; // all current pattern collections
		  ret=chosen_pattern_collections.insert(*pattern_collection); // all current pattern collections
		  if(ret.second==false){
		  DEBUG_MSG(cout<<"pattern_collection is duplicated, skipping"<<endl;);
		  cout<<"pattern_collection is duplicated, skipping"<<endl;
		  pattern_valid=false;
		  fitness = 0.001;
		  collection_counter++;
		  fitness_values.push_back(fitness);
		  continue;
		  }*/

		valid_pattern_counter++;
		if(valid_pattern_counter%100==0){
		  cout<<"time:"<<utils::g_timer()<<",valid_pattern_counter:"<<valid_pattern_counter<<endl;
		}
		DEBUG_MSG(cout<<"pattern valid!,SS evaluating:"<<endl;);
		if(genetic_SS_timer->is_expired()||(double(utils::get_peak_memory_in_kb())/1024>2000)){
		    if(double(utils::get_peak_memory_in_kb())/1024>2000){
			cout<<"no more PDB generation, Peak memory above 2 GB max:"<<utils::get_peak_memory_in_kb()<<endl;
		    }
		    cout<<"breaking-1 out of GA Algortihm, current gen_time:"<<genetic_SS_timer<<" bigger than time_limit:"<<genetic_time_limit<<endl;
		    break;
		}
		avg_sampled_states=double(overall_sampled_states)/double(total_online_samples);
		//cout<<"episode:,"<<current_episode<<",time:,"<<utils::g_timer()<<",overall_pdb_gen_time:,"<<overall_pdb_gen_time<<",overall_pdb_helper_time:,"<<overall_pdb_helper_gen_time<<",online_samples:,"<<total_online_samples<<",overall_sampling_time:,"<<overall_online_samp_time<<",avg samp time:,"<<double(overall_online_samp_time)/double((total_online_samples == 0) ? 1 : total_online_samples)<<",overall_backward_sampling_time:,"<<overall_backward_sampling_timer<<",avg_sampled_states:,"<<avg_sampled_states<<endl;
		//cout<<"Pattern valid,episode:,"<<current_episode<<",time:,"<<utils::g_timer()<<",overall_pdb_gen_time:,"<<overall_pdb_gen_time<<",avg_sampled_states:"<<avg_sampled_states<<endl;
		/* Generate the pattern collection heuristic and get its fitness
		   value. */
		//ZeroOnePDBs zero_one_pdbs(task_proxy, *pattern_collection, *pdb_factory );
		double temp=utils::g_timer();
		cout<<"generating candidate,time:"<<utils::g_timer()<<",size:"<<overall_pdb_size<<endl;
		//cout<<"pattern_collection:"<<*pattern_collection<<endl;fflush(stdout);

		ZeroOnePDBs candidate (task_proxy, *pattern_collection, *pdb_factory, time_limit);
		if(pdb_factory->is_solved()){
		    problem_solved_while_pdb_gen=true;
		    cout<<"Solution found while generating PDB candidate of type:"<<pdb_factory->name()<<", adding PDB and exiting generation at time"<<utils::g_timer()<<endl;

		    cout<<"final episode:,"<<current_episode<<",time:,"<<utils::g_timer()<<",overall_pdb_gen_time:,"<<overall_pdb_gen_time<<",online_samples:,"<<total_online_samples<<",overall_sampling_time:,"<<overall_sampling_time<<",avg samp time:,"<<double(overall_sampling_time)/double((total_online_samples == 0) ? 1 : total_online_samples)<<",avg_sampled_states:,"<<avg_sampled_states<<",overall_probe_time:"<<overall_probe_time<<",candidate_count:,"<<candidate_count<<",unique_samples.size:,"<<unique_samples.size()<<",best_heuristics count:,"<<best_pdb_collections.size()<<endl;
		    best_pdb_collections.push_back(make_shared<PDBCollection> (candidate.get_pattern_databases()));
		    best_pdb_added=true;
		}
		cout<<"finshed generating candidate,time:"<<utils::g_timer()<<",size:"<<overall_pdb_size<<endl;
		candidate_count++;
		//ZeroOnePDBs candidate_explicit(task_proxy, *pattern_collection, *pdb_type_explicit );
		//cout<<"ZeroOnePDBs candidate_explicit has type:"<<pdb_type_explicit->name()<<endl;
		overall_pdb_gen_time+=utils::g_timer()-temp;
		double pdb_gen_time=utils::g_timer()-temp;
		//cout<<"generated candidate,pdb_size:,"<<overall_pdb_size<<",pdb_gen_time:,"<<pdb_gen_time<<endl;
		if(pdb_factory->name()=="symbolic"){
		  if(pdb_gen_time>time_limit){
		    last_sampler_too_big=true;
		    pdb_max_size=max(10000.0,pdb_max_size/10.0);
		    cout<<"Fixing pdb_max_size to:"<<pdb_max_size<<endl;
		  }
		}


		fitness=0.001;
		best_fitness=0.001;
		raised_states=0;
		//int lowered_states=0;
		sampled_states=0;
		//double online_sampling_time=0;
		static int best_initial_value=-1; // Alvaro: Why this is static??????
		//float avg_h_increase=0;
		//states_to_raise=100;
    
		//double probes_start_timer=utils::g_timer();
		//double avg_probe_deviation=0;
		//double avg_probe_result=0;

		const State &initial_state = task_proxy.get_initial_state();
		//DEBUG_MSG(cout<<"\tcurrent_episode:"<<current_episode<<",pdb_max_size:"<<pdb_max_size<<",candidate initial h:"<<candidate.get_value(initial_state)<<endl;fflush(stdout););
		bool run_again=false;
		if(best_pdb_collections.size()==0||create_perimeter){
		    cout<<"no initial heuristic yet"<<endl;fflush(stdout);
		    fitness = 1.0;//best_heuristic not populated yet
		    best_patterns = pattern_collection;
		    if(!create_perimeter){
		      //cout<<"added initial best_pdb"<<endl;
		      best_pdb_collections.push_back(pdb_factory->terminate_creation(candidate.get_pattern_databases()));
		      cout<<"best_pdb_collections.size:"<<best_pdb_collections.size()<<flush<<endl;
			
		      result=make_shared<PatternCollectionInformation>(task, make_shared<PatternCollection>());
		      std::shared_ptr<PDBCollection> best_pdb=make_shared<PDBCollection>  (candidate.get_pattern_databases());
		      if(!best_pdb){
			cout<<"best_pdb is empty!"<<endl;
		      }
		      cout<<"initial h value:"<<candidate.get_value(initial_state)<<flush<<endl;
		      //result->include_additive_pdbs(best_pdb);
			
		      //cout<<"calling recompute"<<flush<<endl;
		      //result->recompute_max_additive_subsets();
		      //best_pdb_collections.resize(1);
		      //best_pdb_collections[0]=result->get_pdbs();
		    }
		    create_perimeter=false;
		    cout<<"initial best_pattern==_collection:"<<endl;
		    //best_patterns->erase(std::remove_if (best_patterns->begin(),best_patterns->end(), delete_empty_vector()),best_patterns->end());
		    for (auto pattern : *best_patterns) {
			cout<<"best_patterns:"<<pattern<<endl;
		    }
		    cout<<"\tbest_heuristic initial h:"<<get_best_value_zero_one(initial_state)<<endl;fflush(stdout);
		    threshold=candidate.get_value(initial_state);
		    continue;
		    //best_patterns = best_pattern_collection;
		} else{
		    //cout<<"current best_heuristics_count:"<<best_heuristic.count_zero_one_pdbs()<<endl;//fflush(stdout);
		    //cout<<"\tbest_heuristic initial h:"<<best_heuristic.get_value(initial_state)<<endl;fflush(stdout);
		    best_heur_dead_ends=0;
		    //double start_sampling_online_time;
		    if(best_initial_value<get_best_value_zero_one(initial_state)){
			best_initial_value=get_best_value_zero_one(initial_state);
			cout<<"timer:,"<<utils::g_timer()<<",best heuristic initial value raised to:"<<best_initial_value<<endl;
			run_again=true;
		    } else if(pruned_states>total_SS_gen_nodes/4){
			run_again=true;
			cout<<"timer:,"<<utils::g_timer()<<",run_again, ratio:"<<total_SS_gen_nodes/pruned_states<<endl;
		    }

		    if(SS_states.size()<5000){
			run_again=true;
			cout<<"Running SS sampling again, SS states left less than 5K"<<endl;
		    }
		    if(run_again){
			SS_states.clear();
			SS_states_vector.clear();
			double start_probe_time=utils::g_timer();
			threshold=max(best_initial_value,max(1,threshold));
			//threshold=44;
			for (int repetition=0;repetition<10;repetition++){
			    vector<double> probe_data;
			    for (int prob_index=0;prob_index<50;prob_index++){
				probe_data.push_back(probe_best_only());
				if(utils::g_timer()-start_probe_time>10.0){
				    cout<<"exceeded 10 seconds limit for probes, number of repetitions completed:"<<repetition<<endl;
				    break;
				} else if(SS_states.size()>20000){
				    cout<<"exceeded 20K max SS_states sampled,current size:"<<SS_states.size()<<endl;
				    break;
				}
			    }
			    pair<double,double> avg_and_dev=utils::avg_and_standard_deviation(probe_data);
			    //cout<<scientific<<"repetition:"<<repetition<<",probe data average:"<<avg_and_dev.first<<endl;
			    //cout<<scientific<<"repetition:"<<repetition<<",probe data standard deviation:"<<avg_and_dev.second<<endl;
			    probe_avgs.push_back(avg_and_dev.first);
		      
			    if(utils::g_timer()-start_probe_time>10.0){
				cout<<"exceeded 10 seconds limit for probes, number of repetions completed:"<<repetition<<endl;
				break;
			    } else if(SS_states.size()>20000){
				cout<<"exceeded 20K max SS_states sampled,current size:"<<SS_states.size()<<endl;
				break;
			    } else if(avg_and_dev.first>pow(10,100)){
				cout<<"avg probe:"<<avg_and_dev.first<<" past 10^100 nodes, not going further, gets very unprecise"<<endl;
				break;
			    }
			    //do not want to go too far, rule of thumb do not increase threshold pass 4 times the initial perimeter distance if it is being used
			    if(initial_perimeter_threshold>0){
			      if(threshold<4*initial_perimeter_threshold)
				threshold=threshold*2;
			    }
			    else{
			      threshold=threshold*2;
			    }

			}
			overall_probe_time+=utils::g_timer()-start_probe_time;
		    
			pair<double,double> avg_and_dev=utils::avg_and_standard_deviation(probe_avgs);
			cout<<"avg probe:"<<avg_and_dev.first<<endl;
			cout<<"avg probe deviation:"<<avg_and_dev.second<<endl;
			cout<<"SS_states.size:"<<SS_states.size()<<endl;
			cout<<"Finished probing with threshold:"<<threshold<<endl;
			//last_sampled_best_heurs_count=best_heuristic.count_zero_one_pdbs();
			cout<<"current_episode:"<<current_episode<<",best_heuristics_count:"<<best_pdb_collections.size()<<",new sampled_states batch"<<endl;fflush(stdout);
			best_heuristic_values.clear();
		  
			cout<<"time:,"<<utils::g_timer()<<",starting sorting SS states,size:"<<SS_states.size()<<endl;
			map<size_t,pair<int,double> >::iterator SS_iter_map;
		
			for(SS_iter_map=SS_states.begin();SS_iter_map!=SS_states.end();){
			    //cout<<",SS_iter_map->first:"<<SS_iter_map->first<<endl;
			    if(unique_samples.find(SS_iter_map->first)==unique_samples.end()) {
				cout<<"state:"<<SS_iter_map->first<<" not in unique_samples!!!"<<endl;exit(0);
			    }
			    State current_state(unique_samples.at(SS_iter_map->first));
			    int current_h=get_best_value_zero_one(current_state);
			    if(current_h==numeric_limits<int>::max()){
				best_heur_dead_ends++;
				SS_iter_map=SS_states.erase(SS_iter_map++);
				//cout<<"eliminating best_heur dead_end, state_id:"<<SS_iter_map->first<<endl;
				continue;
			    }

			    SS_state temp;
			    temp.id=SS_iter_map->first;
			    //cout<<"temp.id:"<<temp.id<<",SS_iter_map->first:"<<SS_iter_map->first<<endl;
			    temp.g=SS_iter_map->second.first;
			    temp.weight=SS_iter_map->second.second;
			    SS_states_vector.push_back(temp);
			    SS_iter_map++;
			}
			sort(SS_states_vector.begin(),SS_states_vector.end(),compare_SS_states);
		    }
		    DEBUG_MSG(cout<<"time:,"<<utils::g_timer()<<",finished randomzing SS states vector,size:"<<SS_states.size()<<",best_heur_dead_ends"<<best_heur_dead_ends<<endl;);
		}
		pruned_states=0;
		fitness=0;
		sampled_states=0;
		raised_states=0;
		double start_sampler_time=utils::g_timer();
		vector<SS_state>::iterator SS_iter;

		DEBUG_MSG(cout<<"SS_states_vector.size:"<<SS_states_vector.size()<<endl;fflush(stdout););
		    
		current_heur_initial_value=candidate.get_value(initial_state);
		total_online_samples++;
		for(SS_iter=SS_states_vector.begin();SS_iter!=SS_states_vector.end();){
		    //cout<<"working on state:"<<SS_iter->id<<endl;
		    if(unique_samples.find(SS_iter->id)==unique_samples.end()){
			cout<<"state not in unique_samples!!!"<<endl;exit(0);
		    }
		    if(sampled_states%100==0){
			if(pruned_states==0){
			    if(utils::g_timer()-start_sampler_time>0.1){
				DEBUG_MSG(cout<<"\tcurrent_episode:"<<current_episode<<",exiting candidate vs best_heuristic SS_states comparison, 0.1 secs iterating without a single better h value"<<endl;);
				break;
			    }
			}
			else if(pruned_states>0){
			    if(utils::g_timer()-start_sampler_time>0.5){
				DEBUG_MSG(cout<<"\t exiting candidate vs best_heuristic SS_states comparison, spent max 0.5 secs"<<endl;);
				break;
			    }
			}
			if(sampled_states>0){
			    if(float(raised_states)/float(sampled_states)>min_improvement_ratio){
				break;
			    }
			}
		    }
		  
		    int best_h=get_best_value_zero_one(unique_samples.at(SS_iter->id));
		    if(best_h==numeric_limits<int>::max()){
			best_heur_dead_ends++;
			SS_states.erase(SS_iter->id);
			SS_iter=SS_states_vector.erase(SS_iter);
			continue;
		    }
		    else if(best_h+SS_iter->g>sampling_threshold){
			SS_states.erase(SS_iter->id);
			SS_iter=SS_states_vector.erase(SS_iter);
			continue;
		    }
		    sampled_states++;
		    overall_sampled_states++;
		    total_SS_gen_nodes+=SS_iter->weight;
		    //cout<<"sampled_state:"<<sampled_states<<",new_f="<<current_heuristic->get_heuristic()+SS_iter->g<<",old f:"<<best_heuristic->get_heuristic()+SS_iter->g<<",g:"<<SS_iter->g<<",weight:"<<SS_iter->weight<<",sampling_threshold:"<<sampling_threshold<<endl;
		    int candidate_h=candidate.get_value(unique_samples.at(SS_iter->id));
		    //cout<<"candidate_h:"<<candidate_h<<",best_h:"<<best_h<<endl;
		    /*if(candidate_h<candidate_explicit.get_value(unique_samples.at(SS_iter->id))){
		      cout<<"candidate_h:"<<candidate_h<<",candidate_explicit:"<<candidate_explicit.get_value(unique_samples.at(SS_iter->id))<<endl;
		      exit(0);
		      }*/
		    if(candidate_h==numeric_limits<int>::max()){
			raised_states++;
			pruned_states+=SS_iter->weight;
			//cout<<"sampled_state:,"<<sampled_states<<",out of "<<SS_states.size()<<"is now pruned by dead_end, weight:"<<SS_iter->weight<<",current_total:"<<total_SS_gen_nodes<<endl;
			SS_iter++;
			continue;
		    }

		    fitness+=candidate_h;

		    if(candidate_h+SS_iter->g>sampling_threshold){
			pruned_states+=SS_iter->weight;
			raised_states++;
			//cout<<"id:,"<<SS_iter->id<<",sampled_state:,"<<sampled_states<<",out of "<<SS_states.size()<<"is now pruned by higher F, weight:"<<SS_iter->weight<<",current_total:"<<total_SS_gen_nodes<<endl;
			//cout<<"h1="<<current_heuristic->get_heuristic()<<"+g="<<SS_iter->g<<",f:"<<current_heuristic->get_heuristic()+SS_iter->g<<",sampling_threshold:"<<sampling_threshold<<endl;
			//cout<<"h2="<<best_heuristic->get_heuristic()<<"+g="<<SS_iter->g<<",f:"<<best_heuristic->get_heuristic()+SS_iter->g<<",sampling_threshold:"<<sampling_threshold<<endl;
		    }
		    else {
			if(candidate_h>best_h){
			    pruned_states+=SS_iter->weight;
			    raised_states++;
			    //cout<<"sampled_state:,"<<sampled_states<<",out of "<<SS_states.size()<<"is now pruned by higher h, weight:"<<SS_iter->weight<<",current_total:"<<total_SS_gen_nodes<<endl;
			}
		    }
		    SS_iter++;
		}
		DEBUG_MSG(cout<<"episode:"<<current_episode<<",finished sampling,sampled_states:"<<sampled_states<<",raised_states:"<<raised_states<<",pruned_states:"<<pruned_states<<endl;fflush(stdout););
		sampler_time=utils::g_timer()-start_sampler_time;
		overall_sampling_time+=sampler_time;
	    DEBUG_MSG(cout<<"sampler_time:"<<sampler_time<<",last_sampler_too_big:"<<last_sampler_too_big<<endl;);
	      
	    if(pdb_factory->name()!="symbolic"){
	      if(!last_sampler_too_big){
		if(sampler_time<2.0&&pdb_gen_time<0.1){
		      last_pdb_max_size=pdb_max_size;
		      last_pdb_min_size=min_size;
		}
		else if(sampler_time>2.0||pdb_gen_time>0.1){
		  cout<<"setting pdb size to :"<<last_pdb_max_size<<",last sampling time with current pdb size took:"<<sampler_time<<",pdb_gen_time:"<<pdb_gen_time<<endl;
		      last_sampler_too_big=true;
		}
	      }
	    }
		DEBUG_MSG(cout<<"collection size:"<<overall_pdb_size<<",sampler_time:,"<<sampler_time<<",candidate_count:,"<<candidate_count<<endl;);
		if(sampled_states>0){
		    fitness/=sampled_states;
		}
		DEBUG_MSG(cout<<"pruned_states="<<pruned_states<<"out of"<<total_SS_gen_nodes<<",avg_h:"<<fitness<<endl;);
		DEBUG_MSG(cout<<"raised_states="<<raised_states<<"out of"<<sampled_states<<",avg_h:"<<fitness<<endl;);
		DEBUG_MSG(cout<<"fitness:"<<fitness<<endl;);
	    //if(sampled_states>0&&float(raised_states)/float(sampled_states)>0)
	      //cout<<"g_timer:,"<<utils::g_timer()<<",current_episode:,"<<current_episode<<",pdb_max_size:,"<<pdb_max_size<<",candidate initial h:,"<<candidate->get_value(initial_state)<<",sampled_states:,"<<sampled_states<<",raised_states:,"<<raised_states<<",ratio:,"<<float(raised_states)/float(sampled_states)<<",overall_pdb_size:,"<<overall_pdb_size<<endl;
		//fitness_values.push_back(fitness);
		if(float(raised_states)/float(sampled_states)>min_improvement_ratio||(best_pdb_collections.size()==0)) {

		    best_fitness = fitness;
		    //if(current_episode>0){
		    //cout << "curr_eps:"<<current_episode<<",improved best_fitness = " << best_fitness << ",";
		    //}

		    //if(best_heuristic.size()!=0){
		    //    delete best_heuristic;
		    // }
		    //else{
		    //  cout<<"best_heuristic being set for the first time"<<endl;
		    //}
		    cout<<"time:,"<<utils::g_timer()<<",bin_packed:,"<<bin_packed_episode<<",adding1 best_heuristic,episode:,"<<current_episode<<",collection:,"<<collection_counter<<",new raised_ratio:,"<<float(raised_states)/float(sampled_states)<<",actual_states_ratio:,"<<float(raised_states)/float(sampled_states)<<",total_nodes:"<<total_SS_gen_nodes<<",pruned_states:"<<pruned_states<<",fitness:,"<<fitness<<",sampled_states:,"<<sampled_states<<",initial_value:,"<<current_heur_initial_value<<",skip_sampling:,"<<skip_sampling<<",best_heur_dead_ends:,"<<best_heur_dead_ends<<",best_heuristics count:"<<best_pdb_collections.size()<<endl;
		    double start_adding_best_time=utils::g_timer();

		    for(size_t i=0;i<pattern_collection->size();i++){
			if(pattern_collection->at(i).size()>0){
			    best_patterns->push_back(pattern_collection->at(i));
			}
		    }
		    best_pdb_collections.push_back(pdb_factory->terminate_creation(candidate.get_pattern_databases()));
		    best_pdb_added=true;
		    if(get_best_value_zero_one(initial_state)<current_heur_initial_value){
			cout<<"DEBUG ME, just added candidate_heur with initial h value:"<<current_heur_initial_value<<"but best_value for initial state is calculated as "<<get_best_value_zero_one(initial_state)<<"!!!!"<<endl;
			exit(1);
		    }

		    double pdb_generation_time=utils::g_timer()-start_adding_best_time;
		    cout<<"time:,"<<utils::g_timer()<<",current_episode:,"<<current_episode<<",best_heuristics count:,"<<best_pdb_collections.size()
			<<",online_sampling_time:,"<<sampler_time<<",pdb_generation_time:,"<<pdb_generation_time<<",pdb_size:,"<<overall_pdb_size
			<<",disjoint:,"<<disjoint_patterns<<",raised:,"<<raised_states<<",disjoint_patterns:,"<<disjoint_patterns
			<<",sampled_states:,"<<sampled_states<<",min_improv_ratio:,"<<min_improvement_ratio<<endl;

		    overall_pdb_gen_time+=pdb_generation_time;

		    DEBUG_MSG(cout<<"overall_pdb_gen_time:"<<overall_pdb_gen_time<<endl;);
		    best_fitness_was_duplicate=false;
		} else {
		    DEBUG_MSG(if(current_heuristic!=NULL){
			cout<<"time:,"<<utils::g_timer()<<",bin_packed:,"<<bin_packed_episode<<",current_heuristic rejected,online_sampling time:,"<<sampler_time
			    <<",raised_ratio:,"<<float(raised_states)/float(sampled_states)<<",fitness:,"<<fitness<<",sampled_states:,"<<sampled_states
			    <<",initial_value:,"<<current_heur_initial_value<<",skip_sampling:,"<<skip_sampling<<",best_heur_dead_ends:,"<<best_heur_dead_ends<<endl;
		    });
		}
	    }
	    collection_counter++;
	    fitness_values.push_back(fitness);
	}
	DEBUG_MSG(cout<<"finished evaluate"<<endl;fflush(stdout););
    }

    void PatternCollectionGeneratorGeneticSS::bin_packing() {
	DEBUG_MSG(cout<<"Starting bin_packing, pdb_max_size:"<<pdb_max_size<<endl;);
	TaskProxy task_proxy(*task);
	VariablesProxy variables = task_proxy.get_variables();

	vector<int> variable_ids;
	variable_ids.reserve(variables.size());
	pattern_collections.clear();
	for (size_t i = 0; i < variables.size(); ++i) {
	    variable_ids.push_back(i);
	}

	for (int i = 0; i < num_collections; ++i) {
	    // Use random variable ordering for all pattern collections.
	    g_rng()->shuffle(variable_ids);
	    vector<vector<bool>> pattern_collection;
	    vector<bool> pattern(variables.size(), false);
	    double current_size = 1;
	    size_t vars_to_combine=variable_ids.size();
	      if(pdb_factory->name()=="symbolic"){
		vars_to_combine = (*g_rng())(variable_ids.size())+1;
	    }
	    DEBUG_MSG(cout<<"1st pattern,vars to combine="<<vars_to_combine<<" out of "<<variable_ids.size()<<endl;);

	    size_t var_counter=0;
	    for (size_t j = 0; j < variable_ids.size(); ++j) {
	      var_counter++;
		int var_id = variable_ids[j];
		double next_var_size = variables[var_id].get_domain_size();

		if (next_var_size > pdb_max_size){
		    //cout<<"\t\tvar:"<<var_id<<" never fits into bin for pdb_max_size:"<<pdb_max_size<<endl;
		    DEBUG_MSG(cout<<"\t\tvar:"<<var_id<<" never fits into bin for pdb_max_size:"<<pdb_max_size<<endl;);
		    // var never fits into a bin.
		    continue;
		}
		else if(!utils::is_product_within_limit(current_size, next_var_size,
							pdb_max_size)) {
			pattern_collection.push_back(pattern);
			//cout<<"\tpattern added to collection, pattern_collection_size:"<<pattern_collection.size()<<endl;
			DEBUG_MSG(cout<<"\tpattern added to collection, pattern_collection_size:"<<pattern_collection.size()<<endl;);
			pattern.clear();
			pattern.resize(variables.size(), false);
			current_size = 1;
			var_counter=0;
			vars_to_combine = (*g_rng())(variable_ids.size()-j)+1;
			DEBUG_MSG(cout<<pattern_collection.size()<<"th pattern,vars to combine="<<vars_to_combine<<" out of remaining "<<variable_ids.size()-j<<endl;);
		}
		else if(pdb_factory->name()=="symbolic"&&vars_to_combine<var_counter){//symbolic pattern, using number of vars instead of pdb_size
		      pattern_collection.push_back(pattern);
		      DEBUG_MSG(cout<<"\t adding pattern["<<pattern_collection.size()-1<<"],pdb_size:"<<current_size*next_var_size<<endl;
		      for(size_t i=0; i<pattern.size(); ++i){
			if(pattern.at(i)){
			      std::cout << i << ',';
			   }
			 }
		      cout<<endl;);
		      pattern.clear();
		      pattern.resize(variables.size(), false);
		      var_counter=0;
		      current_size = 1;
		      vars_to_combine = (*g_rng())(variable_ids.size()-j)+1;
		      DEBUG_MSG(cout<<pattern_collection.size()<<"th pattern,vars to combine="<<vars_to_combine<<" out of remaining "<<variable_ids.size()-j<<endl;);
		}

		  current_size *= next_var_size;
		  DEBUG_MSG(cout<<"\t\tcurrent_size:"<<current_size<<",var_counter:"<<var_counter<<endl;);
		  pattern[var_id] = true;
		  
		  if(pattern_collection.size()>4){
		    break;
		  }
		  //cout<<"\t\tcurrent_size:"<<current_size<<",added var:"<<var_id<<",domain_size of var:"<<next_var_size<<endl;
		}
	    /*
	      The last bin has not bin inserted into pattern_collection, do so now.
	      We test current_size against 1 because this is cheaper than
	      testing if pattern is an all-zero bitvector. current_size
	      can only be 1 if *all* variables have a domain larger than
	      pdb_max_size.
	    */
	    if (current_size > 1 || var_counter>0) {
		pattern_collection.push_back(pattern);
	    }
	    //else{
		//cout<<"\t skipping pattern"<<endl;
	    //}
	    //Sort patterns by size, so zero_one cost partition benefits larger patterns over shorter ones
	    sort(pattern_collection.begin(),pattern_collection.end(),compare_pattern_length);
	    
	    DEBUG_MSG(
		cout<<"\t sorted pattern lengths:";
		for(size_t i=0;i<pattern_collection.size();i++){
		  cout<<std::count(pattern_collection.at(i).begin(),pattern_collection.at(i).end(),true)<<",";
		}
	       	cout<<endl;
	    );
	    
	    pattern_collections.push_back(pattern_collection);
	}
    }

    void PatternCollectionGeneratorGeneticSS::genetic_algorithm(
	shared_ptr<AbstractTask> task_) {
	int time_to_clean_dom=1;
	bin_packed_episode=true;
	task = task_;
	best_fitness = -1;
	best_patterns = nullptr;
	    
       	if(create_perimeter){
	  pattern_collections.clear();
	  TaskProxy task_proxy(*task);
	  VariablesProxy variables = task_proxy.get_variables();
	  Pattern pattern;
	  for(size_t i=0;i<variables.size();i++){
	    pattern.push_back(i);
	  }
	  vector<Pattern> pattern_collection;
	  pattern_collection.push_back(pattern);
	  cout<<"g_timer before calling ZeroOnePDB to generate initial perimeter:"<<utils::g_timer()<<endl;
	  //ZeroOnePDBs *candidate= new ZeroOnePDBs(task_proxy, pattern_collection, *pdb_factory, 10);
	  ZeroOnePDBs candidate(task_proxy, pattern_collection, *pdb_factory, 20);
	  cout<<"g_timer after calling ZeroOnePDB to generate initial perimeter:"<<utils::g_timer()<<endl;
	  cout<<"g_timer before calling terminate_creation to push perimeter into best_pdb_collections"<<utils::g_timer()<<endl;
	  best_pdb_collections.push_back(pdb_factory->terminate_creation(candidate.get_pattern_databases()));
	  best_pdb_added=true;
	  cout<<"g_timer after calling terminate_creation to push perimeter into best_pdb_collections"<<utils::g_timer()<<endl;
	  const State &initial_state = task_proxy.get_initial_state();
	  initial_perimeter_threshold=candidate.get_value(initial_state)*2;
	}
	else{
	  bin_packing();
	}
	vector<double> initial_fitness_values;
	evaluate(initial_fitness_values);
	for (int i = 0; i < num_episodes; ++i) {

	    if(problem_solved_while_pdb_gen){
		cout<<"problem solved, exiting episode loop"<<endl;
		break;
	    }
	    if(genetic_SS_timer->is_expired()){
		//cout<<"breaking-3 out of GA Algortihm, current gen time:"<<timer()<<" bigger than time_limit:"<<time_limit<<endl;
		avg_sampled_states=double(overall_sampled_states)/double(total_online_samples);
		//cout<<"final episode:,"<<current_episode<<",time:,"<<utils::g_timer()<<",overall_pdb_gen_time:,"<<overall_pdb_gen_time<<",overall_pdb_helper_time:,"<<overall_pdb_helper_gen_time<<",online_samples:,"<<total_online_samples<<",overall_sampling_time:,"<<overall_online_samp_time<<",avg samp time:,"<<double(overall_online_samp_time)/double((total_online_samples == 0) ? 1 : total_online_samples)<<",avg_sampled_states:,"<<avg_sampled_states<<",overall_probe_time:"<<overall_probe_time<<endl;
		cout<<"final episode:,"<<current_episode<<",time:,"<<utils::g_timer()<<",overall_pdb_gen_time:,"<<overall_pdb_gen_time<<",online_samples:,"<<total_online_samples<<",overall_sampling_time:,"<<overall_sampling_time<<",avg samp time:,"<<double(overall_sampling_time)/double((total_online_samples == 0) ? 1 : total_online_samples)<<",avg_sampled_states:,"<<avg_sampled_states<<",overall_probe_time:"<<overall_probe_time<<",candidate_count:,"<<candidate_count<<",unique_samples.size:,"<<unique_samples.size()<<",best_heuristics count:,"<<best_pdb_collections.size()<<endl;
		cout<<"Peak memory:"<<utils::get_peak_memory_in_kb()<<endl;fflush(stdout);

	    }
	    else if(current_episode%100==0&&i>0){
	      if(utils::g_timer()>time_to_clean_dom){
		    cout<<"time:"<<utils::g_timer()<<",time to clear dominated heuristics every 100 secs"<<endl;
		    clear_dominated_heuristics();
		    cout<<"time:"<<utils::g_timer()<<",finished clearing dominated heuristics every 100 secs"<<endl;
		    time_to_clean_dom+=100;
		}
		disjoint_patterns=!disjoint_patterns;
		num_collections=1;
		bin_packing();
		bin_packed_episode=true;
		mutation_probability=((double) rand() / (RAND_MAX))/10.0;
	    }
	    else{
		int mutations=mutate2();
		if(mutations==0){
	    DEBUG_MSG(cout<<"mutations=0, next episode"<<endl<<flush;);
		    //cout<<"no mutations, next episode"<<endl;
		    continue;//no mutations!
		}
		bin_packed_episode=false;
	    }

	    if (recompute_max_additive_subsets&&best_pdb_collections.size()>1){  
		/*  cout<<"recomputing best_pdbs as max_additive_subsets,time:"<<utils::g_timer()<<",best collections:"<<best_pdb_collections.size()<<flush<<endl;
		for (auto pdb_collection : best_pdb_collections){
		  result->include_additive_pdbs(pdb_collection);
		}
		cout<<"calling recompute_max_additive_subsets"<<flush<<endl;
		result->recompute_max_additive_subsets();
		best_pdb_collections.resize(1);
		best_pdb_collections[0]=result->get_pdbs();
		time_to_clean_dom+=1;
		best_pdb_added=false;*/
	    }
	  //cout << endl; //cout << "--------- episode no " << (i + 1) << " ---------" << endl;
	    //mutate();
	    vector<double> fitness_values;
	    evaluate(fitness_values);
	    // We allow to select invalid pattern collections.
	    if(genetic_SS_timer->is_expired()||(double(utils::get_peak_memory_in_kb())/1024>2000)){
		avg_sampled_states=double(overall_sampled_states)/double(total_online_samples);
		if (!recompute_max_additive_subsets) {
		  clear_dominated_heuristics();
		}
		//cout<<"final episode:,"<<current_episode<<",g_time:,"<<utils::g_timer()<<",genetic_SS_timer:"<<genetic_SS_timer<<",overall_pdb_gen_time:,"<<overall_pdb_gen_time<<",overall_pdb_helper_time:,"<<overall_pdb_helper_gen_time<<",online_samples:,"<<total_online_samples<<",overall_sampling_time:,"<<overall_online_samp_time<<",avg samp time:,"<<double(overall_online_samp_time)/double((total_online_samples == 0) ? 1 : total_online_samples)<<",overall_backward_sampling_time:,"<<overall_backward_sampling_timer<<",avg_sampled_states:,"<<avg_sampled_states<<endl;
		cout<<"final episode:,"<<current_episode<<",time:,"<<utils::g_timer()<<",overall_pdb_gen_time:,"<<overall_pdb_gen_time<<",online_samples:,"<<total_online_samples<<",overall_sampling_time:,"<<overall_sampling_time<<",avg samp time:,"<<double(overall_sampling_time)/double((total_online_samples == 0) ? 1 : total_online_samples)<<",avg_sampled_states:,"<<avg_sampled_states<<",overall_probe_time:"<<overall_probe_time<<",candidate_count:,"<<candidate_count<<",unique_samples.size:,"<<unique_samples.size()<<",best_heuristics count:,"<<best_pdb_collections.size()<<endl;
		//cout<<"breaking-4 out of GA Algortihm, current gen time:"<<timer()<<" bigger than time_limit:"<<time_limit<<endl;fflush(stdout);
		//clear_dominated_heuristics(&unique_samples);
		break;
	    }
	    //cout<<"calling select"<<endl;fflush(stdout);
	    select(fitness_values);
	    //cout<<"after select"<<endl;fflush(stdout);
	    current_episode++;
	}
	//cout<<"SKIPPING clear_dominated, heurs"<<best_heuristic.count_zero_one_pdbs()<<endl;fflush(stdout);
	
	if (!recompute_max_additive_subsets) {
	  cout<<"starting clear_dominated,starting heurs"<<best_pdb_collections.size()<<endl;fflush(stdout);
	  clear_dominated_heuristics();
	  cout<<"finished clear_dominated,remaining heurs"<<best_pdb_collections.size()<<endl;fflush(stdout);
	}
    }

    double PatternCollectionGeneratorGeneticSS::probe_best_only(){
	DEBUG_MSG(cout<<"calling probe_best_only,threshold:"<<threshold<<endl;fflush(stdout););
  
	//set<int> visited_states;//for cutting off zero-cost operator loops
	TaskProxy task_proxy(*task);
	SuccessorGenerator successor_generator(task);
	const State &initial_state = task_proxy.get_initial_state();

	Options temp_options2;
	//temp_opts.set<shared_ptr<AbstractTask>>(
	//    "transform", task);
	temp_options2.set<int>(
	    "cost_type", NORMAL);
	temp_options2.set<bool>(
	    "cache_estimates", false);
	blind_search_heuristic::BlindSearchHeuristic temp_blind_heuristic(temp_options2);
	sampler = new TypeSystem(&temp_blind_heuristic);
	//cout<<"after sampler"<<endl;fflush(stdout);
	map<Type, SSNode> queue;
	int initial_value=0;
	static int call_number=0;
	call_number++;
	//vector<double> vweight;
	//std::map<Node2, double> expanded;
	//std::map<Node2, double> generated;
	//Heuristic* heuristic;
	set<vector<int> > F_culprits;
	//bool domination_check=false;
	int initial_h=0;
	//int next_f_bound=INT_MAX/2;
	
	//Timer sampling_time;
	/*
	 * Probing is done based on the types of the children of the root state
	 */

        queue.clear();
	//initial_value =current_heur->compute_heuristic(g_initial_state());
	//Using lmcut to define threshold
  
			
        if(best_pdb_collections.size()>0){
	    initial_h=get_best_value_zero_one(initial_state);
	    //initial_h needs to be at least 1
	    initial_h=max(1,initial_h);
	    //initial_h=temp_lmcut_heuristic.compute_heuristic(initial_state);
	    //cout<<"prev_heur_initial_value:"<<initial_value<<endl;
	}
	else{
	    cout<<"cant call probe_best_only if best_heuristic is empty,DEBUG ME!!!"<<endl;
	    exit(0);
	}	  
	//cout<<"initial_GA_h:"<<current_heur->compute_heuristic(g_initial_state())<<endl;
	//threshold = 8 * initial_h;
	//sampling_threshold=max(threshold,sampling_threshold);
	//threshold = 120;
	//threshold = 40;
	//cout<<"threshold="<<threshold<<endl;fflush(stdout);
	sampling_threshold=threshold;
        

	vector<OperatorProxy> applicable_ops; 
	successor_generator.generate_applicable_ops(initial_state,applicable_ops); //count nodes generated
	//cout<<"before amount_initial"<<endl;fflush(stdout);
        double amount_initial = (double)applicable_ops.size();
        //Need to initialize child state before using	
	State child(initial_state); 


 	prev_current_collector=1 + amount_initial;
 	max_collector=1 + amount_initial;
	//cout<<"max_collector:"<<max_collector<<endl;fflush(stdout);
        
	//const State &initial_state2 = g_initial_state();
        size_t initial_state_id = initial_state.hash();
	unique_samples.insert(make_pair(initial_state_id,initial_state));
	//visited_states.insert(initial_state.hash());
	map<size_t,int> cycle_check;
	cycle_check[initial_state.hash()]=0;
	//cout<<"initial_state.hash:"<<initial_state_id<<endl;

	
	SSNode node;

	node.setId(initial_state_id);
	//cout<<"node.ID="<<node.get_id()<<endl;
        node.setWeight(1.0);
        node.setGreal(0);  //g_real value of the node
        //node.setHC(h_initial_v);
        node.setH(initial_value);
	
	SS_states[initial_state_id].first=0;
	SS_states[initial_state_id].second=amount_initial;
	/*
	 * Seeding the prediction with the children of the start state
	 *
	 */
		  

	//Only evaluating those heuristics which still have not pruned the current path
	//cout<<"before getType"<<endl;fflush(stdout);
	Type type = sampler->getType(initial_value);
	//cout<<"after getType"<<endl;fflush(stdout);

	type.setLevel( 0 ); // level where the node is located

	queue.insert( pair<Type, SSNode>( type, node ) );

        int nraiz = 0;
  
	long queue_counter=0;
	//cout<<"before while"<<endl;fflush(stdout);
	//cout<<"queue.size:"<<queue.size()<<endl;fflush(stdout);
	while( !queue.empty() )
	{
	    queue_counter++;
	    if(queue_counter%1000==0){
		if(utils::g_timer()>genetic_time_limit){
		    cout<<"Search_timer past maximum sampling_time"<<endl;fflush(stdout);
		    //cout<<"selecting best heuristic after search_time: "<<search_time()<<", seconds,g_timer:"<<g_timer()<<endl;
		    return(-1);
		}
	    }
#ifdef _SS_DEBUG
	    cout<<"queue.size:"<<queue.size()<<endl;//",search_time:"<<search_time<<endl;
#endif
	    Type out = queue.begin()->first;
	    SSNode s = queue.begin()->second;

	    int g_real =  s.getGreal();
	    int level = out.getLevel();
	    double w = s.getWeight(); 
	    //std::vector<int> h_global_v = s.getHC();
	    //int max_h=s.getH();
                
	    std::map<Type, SSNode>::iterator rt;
	    rt = queue.find(out);


	    queue.erase( rt );
                   
	    nraiz++;                
#ifdef _SS_DEBUG
	    cout<<"Raiz: "<<nraiz<<" h  = "<<max_h;fflush(stdout);
	    cout<<", g_real = "<<g_real<<", f = ";fflush(stdout);
	    cout<<max_h + g_real;fflush(stdout);
	    cout<<", level = "<<level;fflush(stdout);
#endif
		
	    //vweight.push_back(s.getWeight());
		
	        
	    //Insert each node.
	    //Node2 node(getMinHeur(h_global_v) + g_real, level);
	    //Node2 node(max_h + g_real, level);
	    //node.setFs(f_global_v);
	    //node.setL(level);
	    //count nodes expanded
	    /*  if ( (max_h + g_real) <= threshold) {
		std::pair<std::map<Node2, double>::iterator, bool> ret0;

		std::map<Node2, double>::iterator it0;

		ret0 = expanded.insert(std::pair<Node2, double>(node, s.getWeight()));
		it0 = ret0.first;

		if (ret0.second) {
		//cout<<"new node expanded is added."<<endl;
		} else {
		//cout<<"node expanded is being updated."<<endl;
		it0->second += s.getWeight();
		//cout<<"it0->second = "<<it0->second<<endl;
		}
		}*/
	    //cout<<"hola"<<endl;fflush(stdout);
	    //else{//Nodes could be added through BFS, need to update F bound accordingly
#ifdef _SS_DEBUG
	    /* ` int prev_f_bound=next_f_bound;
	       #endif
	       next_f_bound=min(next_f_bound,max_h + g_real);
	       #ifdef _SS_DEBUG
	       if(next_f_bound!=prev_f_bound){
	       cout<<"next_f_bound:"<<next_f_bound<<",prev_f_bound:"<<prev_f_bound<<endl;
	       }*/
#endif
	    //}
	    //end count nodes expanded
	    vector<OperatorProxy> applicable_ops;
	    map<size_t,State>::iterator it=unique_samples.find(s.get_id());
	    if (it == unique_samples.end()){
		cout<<"any retrieved state should be on the queue, FIX ME!!!!"<<endl;exit(0);
	    }

	    State* current_state=&(it->second);
	    successor_generator.generate_applicable_ops(*current_state,applicable_ops); //count nodes generated

	    // std::pair<std::map<Node2, double>::iterator, bool> ret;
	    // std::map<Node2, double>::iterator it;
		

	    /*  ret = generated.insert(std::pair<Node2, double>(node, amount*w));
                it = ret.first;


                if (ret.second) {
		//cout<<"new node is added."<<endl;fflush(stdout);
                } else {
		//cout<<"old is being updated."<<endl;fflush(stdout);
		it->second += amount*w;
		//cout<<"new = "<<it->second<<endl;
                }*/
	    //end count nodes generated
#ifdef _SS_DEBUG
	    cout<<"\t_____________________begin Childs________________________\n";fflush(stdout);
#endif
	    int h =  INT_MAX/2;
	    L.clear();
	    check.clear();
		
	    for (size_t i = 0; i < applicable_ops.size(); ++i)
	    {
		const OperatorProxy &op = applicable_ops[i];
		child = current_state->get_successor(op);
		//  if(visited_states.find(child.hash())!=visited_states.end()){
		//   continue;
		// }

			
		size_t child_hash=child.hash();
		//check if we already have state
		std::map<size_t, int>::iterator cycle_check_iterator=cycle_check.find(child_hash);
			
		//Cycle check
		if(cycle_check.find(child_hash)!=cycle_check.end()){
		    //check if dup state is at lower depth than recorded, otherwise, is cycle so continue
		    if(cycle_check_iterator->second>g_real+op.get_cost()){
			cycle_check_iterator->second=g_real+op.get_cost();
		    }
		    else{
			continue;
		    }
		}
		else{//add state
		    cycle_check[child_hash]=g_real+op.get_cost();
		}

		//SS_states is global to all runs, used here to avoid opening paths where g is known to be suboptimal
		//cycle_check is just to avoid cycles, specially zero-op cost
		if(SS_states.find(child_hash)!=SS_states.end()){
		    if(g_real+op.get_cost()>SS_states[child_hash].first){
			continue;
		    }
		}
		//vector<int> h_child_v;
		//boost::dynamic_bitset<> b_child_v(heuristics.size()+lmcut_heuristic.size()+ipdb_heuristics.size());b_child_v.set();

//                  	vector<int> F_culprit;
			    
		h=get_best_value_zero_one(child);
		//h=temp_lmcut_heuristic.compute_heuristic(child);
		if(h==numeric_limits<int>::max()){
		    continue;
		    //cout<<"prev_h_deade_end_found"<<endl;
		}
		//max_h = max(h, current_heur[*it]->compute_heuristic());
#ifdef _SS_DEBUG
		//cout<<"h("<<i<<"):"<<current_heur->compute_heuristic()<<",f:"<< current_heur->compute_heuristic() + g_real + get_adjusted_cost(*op)<<",thresh:"<<threshold<<endl;
		//int prev_f_bound=next_f_bound;
#endif
			
#ifdef _SS_DEBUG
		/*  if(next_f_bound!=prev_f_bound){
		    cout<<"prev_f_bound:"<<prev_f_bound<<",next_f_bound:"<<next_f_bound<<endl;
		    }*/
#endif
#ifdef _SS_DEBUG
		cout<<", g_real = "<<g_real + get_adjusted_action_cost(*op,cost_type)<<" f_min = "<< h + g_real + get_adjusted_action_cost(*op,cost_type)<<",b_child_v.count:"<<b_child_v.count()<<endl;
		cout<<"\tget_adjusted_cost(*op) = "<<get_adjusted_action_cost(*op,cost_type)<<"\n";
		cout<<"\tChild_"<<(i+1)<<" : h = "<<h<<",b_child_v:"<<b_child_v<<endl; 
		/*  for (size_t i = 0; i < h_child_v.size(); i++) {
		    int h_value = h_child_v.at(i);
		    cout<<h_value + g_real + get_adjusted_cost(*op);
		    if (i != h_child_v.size() -1) {
		    cout<<"/";
		    }
		    }*/
		cout<<h + g_real + get_adjusted_action_cost(*op,cost_type)<<endl;
		cout<<", level = "<<(level + 1);
		cout<<", w = "<<w<<"\n";
#endif

		vector<OperatorProxy> applicable_ops2; 
		//cout<<"S:"<<endl;global_state_2.dump_inline();fflush(stdout);
		successor_generator.generate_applicable_ops(child,applicable_ops2); //count nodes generated
             
		int amount = applicable_ops2.size();
                
		//std::pair<std::map<boost::dynamic_bitset<>, double>::iterator, bool> ret2;
          
		//std::map<boost::dynamic_bitset<>, double>::iterator it2; 
      
		//add to the collector
	
		unique_samples.insert(make_pair(child_hash,child));
		//visited_states.insert(child.hash());
		if ( h + g_real + op.get_cost()  <= threshold) {
		    //Keep a record of all sampled states and their maximum weights and minimum depths
		    if(SS_states.find(child_hash)!=SS_states.end()){
			if(g_real+op.get_cost()<SS_states[child_hash].first){
			    max_collector+=amount*w-SS_states[child_hash].second;
			    //cout<<"Reviewed Stored id:,"<<child_hash<<",new_g:"<<g_real + op.get_cost()<<",new F:"<<h + g_real + op.get_cost()<<endl;
			    SS_states[child_hash].first=g_real + op.get_cost();
			    //If state is found in different probes with same depth, we want to keep record of maximum impact
			    SS_states[child_hash].second=max(SS_states[child_hash].second,amount*w);
			}
			else if(g_real+op.get_cost()<SS_states[child_hash].first){
			    max_collector+=SS_states[child_hash].second;
			}
		    }
		    else{
			max_collector+=amount*w;
			SS_states[child_hash].first=g_real + op.get_cost();
			SS_states[child_hash].second=amount*w;
			//cout<<"New Stored id:,"<<child_hash<<",new_g:"<<g_real + op.get_cost()<<",new F:"<<h + g_real + op.get_cost()<<endl;
		    }

		    //cout<<"hola3"<<endl;fflush(stdout);
		    /*  ret2 = collector.insert(std::pair<boost::dynamic_bitset<>, double>(b_child_v, amount*w));
			it2 = ret2.first;

			if (ret2.second) {
			//cout<<"raiz bc new is added"<<endl;
			} else {
			//cout<<"raiz bc old is being updated"<<endl; 
			it2->second += amount*w;
			//cout<<", newcc : "<<it2->second<<"\n"; 
			}*/

		    //Make pruning
		    Type object = sampler->getType(h);
			   
		    object.setLevel( level + 1 );
                           
		    SSNode child_node;
                           
		    child_node.setId(child_hash);
		    child_node.setWeight(w);
		    child_node.setGreal(g_real + op.get_cost()); 
		    //child_node.setHC(h_child_v);
		    child_node.setH(h);

				
#ifdef _SS_DEBUG
		    cout<<"\t\tChild f<=threshold: h = "<<h; 
		    cout<<", g_real = "<<g_real + get_adjusted_action_cost(*op,cost_type)<<" f = ";
		    cout<<h + g_real  +  get_adjusted_action_cost(*op,cost_type);
		    cout<<", level = "<<level + 1<<"\n";
#endif
		    //ZERO COST OPERATORS NEED EXTRA BFS, until all descendant nodes found for this F-level  
		    //if (op.get_cost() == 0) {//ZERO COST OPERATORS
		    //  cout<<"TO-Do:Update BFS to current version so we can add SS_States to zero-cost operations!"<<endl;exit(0);
//#ifdef _SS_DEBUG
//				cout<<"\t\tget_adjusted_cost(*op) == 0\n";fflush(NULL);
//#endif
//				//TO-DO:UPDATE BFS function to current FD version
//			   	BFS(child_node, object,best_heuristic);
//#ifdef _SS_DEBUG
//				cout<<"after BFS, L size:"<<L.size()<<endl;
//#endif
//				std::set<SSQueue>::iterator it;
//				for (it = L.begin(); it != L.end(); it++) {	
//					SSQueue sst = *it;
//					SSNode node = sst.getNode();
//					Type t = sst.getT();
//					//double new_g_real = node.getGreal();
//					int new_state_id = node.get_id();
//					pair<set<StateID>::iterator, bool> ret2;
//					//std::pair<std::map<boost::dynamic_bitset<>, double>::iterator, bool> ret2;
//					State global_state3 = g_state_registry->lookup_state(new_state_id);
//					unique_samples.insert(global_state3);
//					ret2=visited_states.insert(new_state_id);
//
//					if(SS_states.find(new_state_id)!=SS_states.end()){
//					  if(g_real<SS_states[new_state_id].first){
//					    //cout<<"Reviewed Stored id:,"<<child.get_id()<<",new_g:"<<g_real + get_adjusted_action_cost(*op,cost_type)<<",new F:"<<h + g_real + get_adjusted_action_cost(*op,cost_type)<<endl;
//					    SS_states[new_state_id].first=g_real+get_adjusted_action_cost(*op,cost_type);
//					    SS_states[new_state_id].second=amount*w;
//					  }
//					} 
//					else{
//					  SS_states[new_state_id].first=g_real+get_adjusted_action_cost(*op,cost_type);
//					  SS_states[new_state_id].second=amount*w;
//					}
//					
//					//Check for state in cyclical path
//					if (!ret2.second) {
//					  continue;//the state was already visited
//					}
//					double w2 = node.getWeight();
//
//					//cout<<"\n\t\tNode restored: h = "<<t.getH()<<", g_real = "<<node.getGreal()<<", f = "<<t.getH() + node.getGreal()<<", level = "<<t.getLevel()<<", w = "<<w2<<"\n";
//                           			
//					map<Type, SSNode>::iterator queueIt = queue.find( t );
//#ifdef _SS_DEBUG
//					cout<<"state_id:"<<new_state_id<<endl;
//#endif
//			   		if( queueIt != queue.end() )
//			   		{
//                                	SSNode snode = queueIt->second;
//
//#ifdef _SS_DEBUG
//                                		cout<<"\t\t\tzc: The duplicate node is: h = "<<queueIt->first.getH()<<", g = "<<snode.getGreal()<<", f = "<< queueIt->first.getH() + snode.getGreal()<<", w = "<<snode.getWeight()<<", level = "<<queueIt->first.getLevel()<<"\n";
//#endif
//                                
//						double wa = (double)snode.getWeight();
//						//snode.setWeight( wa + w);
//                                		queueIt->second.setWeight(wa + w2); // set w the node that already exists
//                                		//cout<<"\t\t\tzc: before ss process starts, the w of the duplicate node is updated to: "<<queueIt->second.getWeight()<<endl; 
//                                		//std::pair<std::map<Type, SSNode>::iterator, bool> ret0;
//                                		//ret0 = queue.insert(pair<Type, SSNode>(object, snode));
//                                		//cout<<"\tsnode.getWeight() = "<<snode.getWeight()<<endl;
//                                		//queueIt->second.setWeight(snode.getWeight());
//						double prob = ( double )w2 / (double)( wa + w2);
//						//int rand_100 =  RanGen2->IRandom(0, 99);  //(int)g_rng.next(100);
//						int rand_100 =  rand()%100;  //(int)g_rng.next(100);
//                          	 
//                                		double a = (( double )rand_100) / 100;
//                                		//cout<<"a = "<<a<<" prob = "<<prob<<endl;
//                                
//						if (a < prob) 
//						{
//#ifdef _SS_DEBUG
//                                        		cout<<"\t\t\tzc: Added even though is duplicate.\n";                               
//#endif
//				        		node.setWeight( wa + w2);
//#ifdef _SS_DEBUG
//                                        		cout<<"\t\t\tzc: the w is updated to = "<<node.getWeight()<<endl;
//#endif
//                                        		std::pair<std::map<Type, SSNode>::iterator, bool> ret3;
//                                     			queue.erase(t); 
//                                        
//                                        		ret3 = queue.insert( pair<Type, SSNode>( t, node ));      
//                                        		queueIt = ret3.first;
//                                        		queueIt->second.setWeight(node.getWeight());
//						} else {
//#ifdef _SS_DEBUG
//                                        		cout<<"\t\t\tzc: Not added.\n";
//                                        		cout<<"\t\t\tbut the w is updated for the node that already exists to: "<<queueIt->second.getWeight()<<endl;
//#endif
//                                		}
//			   		} 
//			   		else
//			   		{
//#ifdef _SS_DEBUG
//                                		cout<<"\t\t\tzc: New L node added.\n";
//#endif
//						queue.insert( pair<Type, SSNode>( t, node ) );
//                                		//cout<<"\t\tsucc_node2.getWeight() = "<<succ_node2.getWeight()<<"\n";
//                                
//                                		//cout<<"\t\t\tzc: Child: h = "<< t.getH() <<", g_real = "<< new_g_real <<", f = "<< t.getH() + new_g_real << " threshold: " << threshold <<" w = "<<node.getWeight()<<endl;
//                           		}// End queueIt != queue.end()
//				}   //End for set lopp
//#ifdef _SS_DEBUG
//				cout<<"Finished with L"<<endl;
//#endif
		    //} else {

						       
		    map<Type, SSNode>::iterator queueIt = queue.find( object );
		    if( queueIt != queue.end() )
		    {
			SSNode snode = queueIt->second;

#ifdef _SS_DEBUG
			cout<<"\t\tThe duplicate node is: h = "<<h;
				      
			cout<<", g_real = "<<g_real + get_adjusted_action_cost(*op,cost_type)<<" f = ";
			cout<<h + g_real  +  get_adjusted_action_cost(*op,cost_type);
			cout<<", w = "<<snode.getWeight();
			cout<<", level = "<<level + 1<<"\n";
#endif

			double wa = (double)snode.getWeight();
			//snode.setWeight( wa + w);
			queueIt->second.setWeight(wa + w);
#ifdef _SS_DEBUG
			cout<<"\t\tbefore ss process starts, the w of the duplicate node is updated to: "<<queueIt->second.getWeight()<<endl;
#endif
			//std::pair<std::map<Type, SSNode>::iterator, bool> ret0;

			//ret0 = queue.insert(pair<Type, SSNode>(object, snode));
			//cout<<"\tsnode.getWeight() = "<<snode.getWeight()<<endl;
			//queueIt->second.setWeight(snode.getWeight());
   
   
			double prob = ( double )w / (double)( wa + w );
			//int rand_100 =  RanGen2->IRandom(0, 99);  //(int)g_rng.next(100);
			int rand_100 =  rand()%100;  //(int)g_rng.next(100);
				   
			double a = (( double )rand_100) / 100;
			//cout<<"a = "<<a<<" prob = "<<prob<<endl; 
				  
			if( (a < prob))
			{
			    //unique_samples.insert(make_pair(child.hash(),child));
			    //cout<<"a<prob,hash:"<<child.hash()<<endl;
#ifdef _SS_DEBUG
			    cout<<"\t\tAdded even though is duplicate.\n";
#endif
					  
			    child_node.setWeight( wa + w);
#ifdef _SS_DEBUG
			    cout<<"\t\tthe w is updated to = "<<child_node.getWeight()<<endl;
#endif
			    std::pair<std::map<Type, SSNode>::iterator, bool> ret;
			    queue.erase(object); 

			    ret = queue.insert( pair<Type, SSNode>( object, child_node ));      

			    queueIt = ret.first;
			    queueIt->second.setWeight(child_node.getWeight());
					  
					  
			} else {
#ifdef _SS_DEBUG
			    cout<<"\t\tNot added.\n";
			    cout<<"\t\tbut the w is updated for the node that already exists to: "<<queueIt->second.getWeight()<<endl;
#endif
			}
		    } 
		    else
		    {
			//unique_samples.insert(make_pair(child.hash(),child));
			//cout<<"new_SS_state,hash:"<<child.hash()<<endl;
#ifdef _SS_DEBUG
			cout<<"\t\tNew node added\n";
			//Now update the non-prunning set of heuristics for the node
#endif
			queue.insert( pair<Type, SSNode>( object, child_node ) );
		    }
		    //}
		}
		else 
		{
#ifdef _SS_DEBUG
		    cout << "\tNode was pruned!" << endl;
#endif
		}
#ifdef _SS_DEBUG
		cout<<"\tend Child_"<<(i+1)<<"\n";
#endif
	    }
	}
	//boost::dynamic_bitset<> max_comb(heuristics.size()+lmcut_heuristic.size()+ipdb_heuristics.size());max_comb.set();
	//boost::dynamic_bitset<> max_comb(1);max_comb.set();
	//cout <<std::scientific<<",max_comb_nodes:"<<collector[max_comb]<<endl;
	//cout <<std::scientific<<",max_comb_nodes:"<<collector<<endl;
	delete sampler;
	//cout<<"\t probe_best_only finished,threshold:"<<threshold<<",max_collector:"<<max_collector<<endl;
	return max_collector;
    }
    double PatternCollectionGeneratorGeneticSS::get_pattern_size(Pattern pattern){
	// test if the pattern respects the memory limit
	double mem = 1;
	for (size_t j = 0; j < pattern.size(); ++j) {
	    double domain_size = g_variable_domain[pattern[j]];
	    mem *= domain_size;
	}   
	return mem;
    }
    int PatternCollectionGeneratorGeneticSS::get_best_value(State current_state){
	int h=result->get_value(current_state);
	//cout<<"h:"<<h<<",best_pdb_collections.size:"<<best_pdb_collections.size()<<flush<<endl;
	
	if(best_pdb_collections.size()==1){
	  return h;
	}
	
	for (size_t collection=1;collection<best_pdb_collections.size(); collection++){
	    int h_part=0;
	    for (auto pdb : *best_pdb_collections[collection]){
		int h_temp=pdb->get_value(current_state);
		if (h_temp == numeric_limits<int>::max()){
		    //cout<<"\th:"<<numeric_limits<int>::max()<<endl;
		    return numeric_limits<int>::max();
		}
		h_part+=h_temp;
		//cout<<"\t\t\th_part:"<<h_part<<",h_temp:"<<h_temp<<endl;
	    }
	    h=max(h,h_part);
	    //cout<<"\t\th:"<<h<<",h_part:"<<h_part<<endl;
	}
	//cout<<"\th:"<<h<<endl;
	return h;
    }
    int PatternCollectionGeneratorGeneticSS::get_best_value_zero_one(State current_state){
	//So all the input best collections are zero_ones!
	int h=0;
	for (size_t collection=0;collection<best_pdb_collections.size(); collection++){
	    int h_part=0;
	    for (auto pdb : *best_pdb_collections[collection]){
		int h_temp=pdb->get_value(current_state);
		if (h_temp == numeric_limits<int>::max()){
		    //cout<<"\th:"<<numeric_limits<int>::max()<<endl;
		    return numeric_limits<int>::max();
		}
		h_part+=h_temp;
		//cout<<"\t\t\th_part:"<<h_part<<",h_temp:"<<h_temp<<endl;
	    }
	    h=max(h,h_part);
	    //cout<<"\t\th:"<<h<<",h_part:"<<h_part<<endl;
	}
	//cout<<"\th:"<<h<<endl;
	return h;
    }


    void PatternCollectionGeneratorGeneticSS::clear_dominated_heuristics(){
	if(best_pdb_collections.size()<2){
	    return;
	}

	double start_time=utils::g_timer();
	cout<<"calling clear_dominated_heuristics with "<<best_pdb_collections.size()<<" best heuristics and unique_samples:"<<unique_samples.size()<<endl;fflush(stdout);
	std::vector<std::shared_ptr<PDBCollection> > cleaned_best_pdb_collections; //Store the PDBs as well

	TaskProxy task_proxy(*task);
	const State &initial_state = task_proxy.get_initial_state();
	//size_t initial_state_id = initial_state.hash();

	vector<int> current_best_h_values;
	int h=0;
	int h_temp=0;
	for(map<size_t,State>::iterator it=unique_samples.begin(); it!=unique_samples.end();it++){
	    //cout<<"before evaluate state_id:"<<it->get_id()<<endl;fflush(stdout);
	    //cout<<"State:";it->dump_inline();cout<<endl;fflush(stdout);
	    h=0;
	    h_temp=0;
	    for (auto pdb : *best_pdb_collections.back()){
		h_temp=pdb->get_value(it->second);
		if (h_temp == numeric_limits<int>::max()){
		    h=numeric_limits<int>::max();
		    break;
		}
		else{
		    h+=h_temp;
		}
	    }


	    //cout<<"h:"<<best_pdb_collections.back()->get_heuristic()<<endl;fflush(stdout);
	    //cout<<"after evaluate"<<endl;fflush(stdout);
        
	    if (h != numeric_limits<int>::max()){
		current_best_h_values.push_back(h);
	    }
	    else{
		current_best_h_values.push_back(INT_MAX);
	    }
	}
    
  cleaned_best_pdb_collections.push_back(best_pdb_collections.back());
  //cout<<"current_best_h_values.size:"<<current_best_h_values.size()<<endl;fflush(stdout);
  for(int i=best_pdb_collections.size()-2;i>=0;i--){
    bool dominated_heur=true;
    //cout<<"i:"<<i<<endl;
    int j=0;
    int h=0;
    int h_temp=0;
    for(map<size_t,State>::iterator it=unique_samples.begin(); it!=unique_samples.end();it++){
      if(current_best_h_values[j]==INT_MAX){
	j++;
	continue;
      }
      h=0;
      h_temp=0;
      for (auto pdb : *best_pdb_collections.at(i)){
	h_temp=pdb->get_value(it->second);
	if (h_temp == numeric_limits<int>::max()){
	  h=numeric_limits<int>::max();
	  break;
	}
	else{
	  h+=h_temp;
	}
      }
      //NO BREAKS BECAUSE WE WANT TO CALCULATE ALL THE NEW HIGHER H VALUES
      //IF HEUR IS NOT DOMINATED
      if (h == numeric_limits<int>::max()){
	dominated_heur=false;
	current_best_h_values[j]=INT_MAX;
      }
      else if(h>current_best_h_values[j]){
	dominated_heur=false;
	current_best_h_values[j]=h;
      }
      j++;
    }
    if(!dominated_heur){
      cout<<"adding heur["<<i<<"] to list of heurs"<<endl;
      cleaned_best_pdb_collections.push_back(best_pdb_collections.at(i));
      /*if(cleaned_best_pdb_collections.size()>15){
	cout<<"max of 15 pdb_collections, otherwise timewise takes too long"<<endl;
       break;
      } */
    }
    else{
      cout<<"collection["<<i<<"] is dominated,eliminating "<<endl;
      best_pdb_collections.at(i).reset();
    }
  }
  cout<<"best_pdb_collections size:"<<best_pdb_collections.size();
  best_pdb_collections=cleaned_best_pdb_collections;
  cout<<",cleaned_best_pdb_collections:"<<cleaned_best_pdb_collections.size()<<","<<best_pdb_collections.size()<<",time:"<<utils::g_timer()-start_time<<endl;
}
//PatternCollectionInformation PatternCollectionGeneratorGeneticSS::generate(
//    shared_ptr<AbstractTask> task) {
//    utils::Timer timer;
//    genetic_algorithm(task);
//    cout << "Pattern generation (Edelkamp) time: " << timer << endl;
//    cout<<"best_patterns.size:"<<best_patterns->size()<<endl;
//    for (auto pattern : *best_patterns) {
//      cout<<"best_patterns:"<<pattern<<endl;
//    }
//    assert(best_patterns);
//    delete genetic_SS_timer;
//    return PatternCollectionInformation(task, best_patterns);
//}
    PatternCollectionInformation PatternCollectionGeneratorGeneticSS::generate(
	shared_ptr<AbstractTask> task) {
	utils::Timer timer;

	genetic_algorithm(task);
    
	PatternCollectionInformation result2 (task, make_shared<PatternCollection>());

      
	int counter1=0;
	for (auto pdb_collection : best_pdb_collections){
	    result2.include_additive_pdbs(pdb_collection);
	    int counter2=0;
	    for (auto pdb : *pdb_collection){
		cout<<"best_pdb_collection #"<<counter1<<"["<<counter2<<"]"<<*pdb<<endl;
		counter2++;
	    }
	    counter1++;
	}

	if (recompute_max_additive_subsets) {
	    result2.recompute_max_additive_subsets();
	}

	result2.set_dead_ends(pdb_factory->get_dead_ends());

	cout <<"Finished,episodes:"<<current_episode<<",Pattern generation (Edelkamp) time: " << timer <<",Peak Memory:"<<utils::get_peak_memory_in_kb()<<endl;fflush(stdout);
	assert(best_patterns);
	return result2;
    }

    static shared_ptr<PatternCollectionGenerator> _parse(OptionParser &parser) {
	parser.document_synopsis(
	    "Genetic Algorithm Patterns",
	    "The following paper describes the automated creation of pattern "
	    "databases with a genetic algorithm. Pattern collections are initially "
	    "created with a bin-packing algorithm. The genetic algorithm is used "
	    "to optimize the pattern collections with an objective function that "
	    "estimates the mean heuristic value of the the pattern collections. "
	    "Pattern collections with higher mean heuristic estimates are more "
	    "likely selected for the next generation." + utils::format_paper_reference(
		{"Stefan Edelkamp"},
		"Automated Creation of Pattern Database Search Heuristics",
		"http://www.springerlink.com/content/20613345434608x1/",
		"Proceedings of the 4th Workshop on Model Checking and Artificial"
		" Intelligence (!MoChArt 2006)",
		"35-50",
		"2007"));
	parser.document_language_support("action costs", "supported");
	parser.document_language_support("conditional effects", "not supported");
	parser.document_language_support("axioms", "not supported");
	parser.document_note(
	    "Note",
	    "This pattern generation method uses the "
	    "zero/one pattern database heuristic.");
	parser.document_note(
	    "Implementation Notes",
	    "The standard genetic algorithm procedure as described in the paper is "
	    "implemented in Fast Downward. The implementation is close to the "
	    "paper.\n\n"
	    "+ Initialization<<BR>>"
	    "In Fast Downward bin-packing with the next-fit strategy is used. A "
	    "bin corresponds to a pattern which contains variables up to "
	    "``pdb_max_size``. With this method each variable occurs exactly in "
	    "one pattern of a collection. There are ``num_collections`` "
	    "collections created.\n"
	    "+ Mutation<<BR>>"
	    "With probability ``mutation_probability`` a bit is flipped meaning "
	    "that either a variable is added to a pattern or deleted from a "
	    "pattern.\n"
	    "+ Recombination<<BR>>"
	    "Recombination isn't implemented in Fast Downward. In the paper "
	    "recombination is described but not used.\n"
	    "+ Evaluation<<BR>>"
	    "For each pattern collection the mean heuristic value is computed. For "
	    "a single pattern database the mean heuristic value is the sum of all "
	    "pattern database entries divided through the number of entries. "
	    "Entries with infinite heuristic values are ignored in this "
	    "calculation. The sum of these individual mean heuristic values yield "
	    "the mean heuristic value of the collection.\n"
	    "+ Selection<<BR>>"
	    "The higher the mean heuristic value of a pattern collection is, the "
	    "more likely this pattern collection should be selected for the next "
	    "generation. Therefore the mean heuristic values are normalized and "
	    "converted into probabilities and Roulette Wheel Selection is used.\n"
	    "+\n\n", true);

	parser.add_option<double>(
	    "pdb_max_size",
	    "maximal number of states per pattern database ",
	    "50000",
	    Bounds("1", "infinity"));
	parser.add_option<int>(
	    "num_collections",
	    "number of pattern collections to maintain in the genetic "
	    "algorithm (population size)",
	    "5",
	    Bounds("1", "infinity"));
	parser.add_option<int>(
	    "num_episodes",
	    "number of episodes for the genetic algorithm",
	    "30",
	    Bounds("0", "infinity"));
	parser.add_option<double>(
	    "mutation_probability",
	    "probability for flipping a bit in the genetic algorithm",
	    "0.01",
	    Bounds("0.0", "1.0"));
	parser.add_option<bool>(
	    "disjoint",
	    "consider a pattern collection invalid (giving it very low "
	    "fitness) if its patterns are not disjoint",
	    "false");
	parser.add_option<bool>(
	    "hybrid_pdb_size",
	    "mix pdb_sizes according to generation time",
	    "true");

    parser.add_option<double>(
        "pdb_max_size",
        "maximal number of states per pattern database ",
        "50000",
        Bounds("1", "infinity"));
    parser.add_option<int>(
        "num_collections",
        "number of pattern collections to maintain in the genetic "
        "algorithm (population size)",
        "5",
        Bounds("1", "infinity"));
    parser.add_option<int>(
        "num_episodes",
        "number of episodes for the genetic algorithm",
        "30",
        Bounds("0", "infinity"));
    parser.add_option<double>(
        "mutation_probability",
        "probability for flipping a bit in the genetic algorithm",
        "0.01",
        Bounds("0.0", "1.0"));
    parser.add_option<bool>(
        "disjoint",
        "consider a pattern collection invalid (giving it very low "
        "fitness) if its patterns are not disjoint",
        "false");
    parser.add_option<shared_ptr<PDBFactory>>(
        "pdb_factory",
        "See detailed documentation for pdb factories. ",
	"symbolic");
    parser.add_option<bool>(
        "recompute_max_additive_subsets",
        "attempts to recompute max additive subsets after generating all patterns",
        "true");
    parser.add_option<int>(
        "time_limit",
        "time limit in seconds for symbolic pdb_generation cut off",
        "1");
    parser.add_option<int>(
        "genetic_time_limit",
        "time limit in seconds for genetic algorithm cut off",
        "900");

	Options opts = parser.parse();
	if (parser.dry_run())
	    return 0;

	return make_shared<PatternCollectionGeneratorGeneticSS>(opts);
    }
    ostream& operator<<(ostream& os, const vector<bool>& v){
	os << "[";
	for(size_t i=0; i<v.size(); ++i){
	    if(v.at(i)){
		os <<i<<",";
	    }
	}
	return os;
    }

    static PluginShared<PatternCollectionGenerator> _plugin("genetic_ss", _parse);
}
