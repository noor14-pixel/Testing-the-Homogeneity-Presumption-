* ============================================================================
* VARIABLE LABELS FOR BETTER INTERPRETATION
* ============================================================================
recode cdisability (2=0)
rename hand_arm_leg hit_hand_arm_leg
* Learning outcomes
label variable learn_letters "Identifies 10+ letters"
label variable read_words "Reads 4+ simple words" 
label variable know_numbers "Recognizes numbers 1-10"
label variable learning_score "Learning Score (0-3)"

* Demographic variables
label variable Female "Female"
label variable Male "Male"
label variable cage "Age (months)"

* Education levels
label variable None "No education"
label variable Primary "Primary education"
label variable Middle "Middle education" 
label variable Secondary "Secondary education"
label variable Higher "Higher education"

* Residence
label variable Rural "Rural residence"
label variable Urban "Urban residence"

* Wealth quintiles
label variable poor "Poorest"
label variable second "Second poorest"
label variable middle "Middle wealth"
label variable fourth "Fourth richest"
label variable richest "Richest"

* School attendance
label variable school "Attended preschool"

* Disciplinary actions
label variable take_away_privileges "Took away privileges"
label variable explain_behaviour "Explained behavior"
label variable shook_child "Shook child"
label variable shouted_yelled_child "Shouted/yelled at child"
label variable gave_something_else "Gave something else to do"
label variable spanked_hit_slapped "Spanked/hit/slapped"
label variable hit_bottom_with_item "Hit with object"
label variable name_calling_dumb_lazy "Called dumb/lazy"
label variable hit_face_head_ears "Hit face/head/ears"
label variable hit_hand_arm_leg "Hit hand/arm/leg"
label variable beat_as_hard_one_could "Beat child hard"

* Household variables
label variable new_variable "Household ID"
label variable children_per_household "Children per household"

* ============================================================================
* MODEL 1: ORDERED LOGIT REGRESSION
* ============================================================================

ologit learning_score poor second middle fourth richest Rural Urban Higher Secondary Middle Primary None Female Male cage ///
    take_away_privileges explain_behaviour shook_child shouted_yelled_child gave_something_else ///
    spanked_hit_slapped hit_bottom_with_item name_calling_dumb_lazy hit_face_head_ears ///
    hit_hand_arm_leg beat_as_hard_one_could school cdisability, ///
    vce(cluster HH1)

* Export ordered logit results
esttab using "table_ologit_coefficients_final_final.tex", replace ///
    b se star(* 0.05 ** 0.01 *** 0.001) ///
    title("Ordered Logit: Determinants of Learning Score") ///
    label booktabs

* Marginal effects for all variables
margins, dydx(*)
estpost margins, dydx(*)
esttab using "table_ologit_margins_all_final_final.tex", replace ///
    cells("b(star) se(par)") ///
    star(* 0.05 ** 0.01 *** 0.001) ///
    title("Marginal Effects: Ordered Logit (All Covariates)") ///
    label booktabs


* Marginal effects for disciplinary actions only
margins, dydx(take_away_privileges explain_behaviour shook_child shouted_yelled_child ///
               gave_something_else spanked_hit_slapped hit_bottom_with_item ///
               name_calling_dumb_lazy hit_face_head_ears hit_hand_arm_leg ///
               beat_as_hard_one_could)

estpost margins, dydx(take_away_privileges explain_behaviour shook_child shouted_yelled_child ///
               gave_something_else spanked_hit_slapped hit_bottom_with_item ///
               name_calling_dumb_lazy hit_face_head_ears hit_hand_arm_leg ///
               beat_as_hard_one_could)

esttab using "table_ologit_margins_disciplinary_final_final.tex", replace ///
    cells("b se p") star(* 0.05 ** 0.01 *** 0.001) ///
    title("Marginal Effects: Ordered Logit (Disciplinary Actions)") ///
    label booktabs

* Plot for disciplinary actions
marginsplot, horizontal ///
    title("Marginal Effects: Disciplinary Actions on Learning Score") ///
    ytitle("Disciplinary Actions") xtitle("Marginal Effect (dy/dx)") ///
    plotopts(msymbol(circle) msize(small)) ///
    scheme(s1color) ///
    xsize(8) ysize(6)
graph export "figure_ologit_margins_disciplinary_final_final.png", replace width(2000) height(1500)

* ============================================================================
* MODEL 2: GENERALIZED ORDERED LOGIT
* ============================================================================

gologit2 learning_score poor second middle fourth ///
    Rural Higher Secondary Primary Female cage cdisability ///
    take_away_privileges explain_behaviour shook_child shouted_yelled_child gave_something_else ///
    spanked_hit_slapped hit_bottom_with_item name_calling_dumb_lazy hit_face_head_ears ///
    hit_hand_arm_leg beat_as_hard_one_could school, vce(cluster HH1) autofit



* Export generalized logit results
esttab using "table_gologit_coefficients_final_final.tex", replace ///
    b se star(* 0.05 ** 0.01 *** 0.001) ///
    title("Generalized Ordered Logit: Learning Score") ///
    label booktabs

* Marginal effects for generalized logit
margins, dydx(*)
estpost margins, dydx(*)
esttab using "table_gologit_margins_all_final_final.tex", replace ///
     cells("b(star) se(par)") ///
    star(* 0.05 ** 0.01 *** 0.001) ///
    title("Marginal Effects: Generalized Ordered Logit") ///
    label booktabs
	
margins, dydx(take_away_privileges explain_behaviour shook_child shouted_yelled_child ///
               gave_something_else spanked_hit_slapped hit_bottom_with_item ///
               name_calling_dumb_lazy hit_face_head_ears hit_hand_arm_leg ///
               beat_as_hard_one_could)
* Large margins plot for all covariates
marginsplot, horizontal ///
    title("Marginal Effects: Generalized Ordered Logit") ///
    ytitle("Disciplinary Actions") xtitle("Marginal Effect (dy/dx)") ///
    plotopts(msymbol(circle) msize(small)) ///
    scheme(s1color) ///
    xsize(10) ysize(12)
graph export "figure_gologit_margins_all_final_final.png", replace width(3000) height(2500)

* ============================================================================
* HOUSEHOLD CLUSTER ANALYSIS
* ============================================================================

* Calculate children per household
bysort new_variable: gen children_per_hh = _N
tabulate children_per_hh
rename new_variable household_number

* ============================================================================
* MODEL 3: MIXED-EFFECTS ORDERED LOGIT
* ============================================================================

meologit learning_score poor second middle fourth richest Rural Urban Higher Secondary Middle Primary None Female Male cage cdisability ///
    take_away_privileges explain_behaviour shook_child shouted_yelled_child gave_something_else ///
    spanked_hit_slapped hit_bottom_with_item name_calling_dumb_lazy hit_face_head_ears ///
    hit_hand_arm_leg beat_as_hard_one_could school ///
    || household_number:, vce(cluster HH1)

* Export mixed-effects results
esttab using "table_meologit_coefficients_final_final.tex", replace ///
    b se star(* 0.05 ** 0.01 *** 0.001) ///
    title("Mixed-Effects Ordered Logit: Learning Score") ///
    label booktabs

* ICC calculation
estat icc
esttab using "table_meologit_icc_final_final.tex", replace ///
    title("Intraclass Correlation: Mixed-Effects Model") ///
    label booktabs
/// something wrong is happening here why am i getting the sameresults lik ein simple ordered logit . please someone try this on their l;aptop
* Marginal effects for disciplinary actions in mixed model
margins, dydx(take_away_privileges explain_behaviour shook_child shouted_yelled_child gave_something_else ///   
    spanked_hit_slapped hit_bottom_with_item name_calling_dumb_lazy hit_face_head_ears ///
    hit_hand_arm_leg beat_as_hard_one_could)

estpost margins, dydx(take_away_privileges explain_behaviour shook_child shouted_yelled_child gave_something_else ///
    spanked_hit_slapped hit_bottom_with_item name_calling_dumb_lazy hit_face_head_ears ///
    hit_hand_arm_leg beat_as_hard_one_could)

esttab using "table_meologit_margins_disciplinary_final_final.tex", replace ///
    cells("b(star) se(par)") ///
    star(* 0.05 ** 0.01 *** 0.001) ///
    title("Marginal Effects: Mixed-Effects Ordered Logit (Disciplinary Actions)") ///
    label booktabs

* Final margins plot for disciplinary actions in mixed model
marginsplot, horizontal ///
    title("Marginal Effects: Mixed-Effects Ordered Logit") ///
    ytitle("Disciplinary Actions") xtitle("Marginal Effect (dy/dx)") ///
    plotopts(msymbol(circle) msize(small)) ///
    scheme(s1color) ///
    xsize(8) ysize(6)
graph export "figure_meologit_margins_disciplinary_final_final.png", replace width(2000) height(1500)

* ============================================================================
* COMPARISON TABLE: ALL MODELS// Not valid
* ============================================================================

* Store estimates for comparison
estimates store ologit_model
estimates store gologit_model  
estimates store meologit_model

* Create comparison table
esttab ologit_model gologit_model meologit_model using "table_model_comparison.tex", replace ///
    b se star(* 0.05 ** 0.01 *** 0.001) ///
    title("Comparison of Ordered Logit Models") ///
    label booktabs mtitle("Ordered Logit" "Generalized OL" "Mixed-Effects OL")
****************************************************************************Verify the Random Forest Results Add Interaction Term**************************************************************
//Adding Interaction Term to the normal oredered logit formula Domestic Violence * Disciplinary Action   Parents Belief about Discipline * Disciplinary Action 
