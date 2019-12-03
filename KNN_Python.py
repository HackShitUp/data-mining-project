import csv
import math

class Person:
    def __init__(self, age, private, self_emp_not_inc, self_emp_inc, federal_gov, local_gov, state_gov, without_pay, never_worked, unknown_workclass, fnlwgt, bachelors, some_college, _11th, hs_grad, prof_school, assoc_acdm, assoc_voc, _9th, _7th_8th, _12th, masters, _1st_4th, _10th, doctorate, _5th_6th, preschool, unknown_education, education_num, married_civ_spouse, divorced, never_married, separated, widowed, married_spouse_absent, married_af_spouse, unknown_marital_status, tech_support, craft_repair, other_service, sales, exec_managerial, prof_specialty, handlers_cleaners, machine_op_inspct, adm_clerical, farming_fishing, transport_moving, priv_house_serv, protective_serv, armed_forces, unknown_occupation, wife, own_child, husband, not_in_family, other_relative, unmarried, unknown_relationship, white, asian_pac_islander, amer_indian_eskimo, other, black, unknown_race, sex, capital_gain, capital_loss, hours_per_week, united_states, cambodia, england, puerto_rico, canada, germany, outlying_us, india, japan, greece, south, china, cuba, iran, honduras, philippines, italy, poland, jamaica, vietnam, mexico, portugal, ireland, france, dominican_republic, laos, ecuador, taiwan, haiti, columbia, hungary, guatemala, nicaragua, scotland, thailand, yugoslavia, el_salvador, trinadad_tobago, peru, hong, holand_netherlands, unknown_country, label):
        self.features = []
        # age (int)
        self.features.append(age)
        # workclass (bool, one hot encoding)
        self.features.append(private)
        self.features.append(self_emp_not_inc)
        self.features.append(self_emp_inc)
        self.features.append(federal_gov)
        self.features.append(local_gov)
        self.features.append(state_gov)
        self.features.append(without_pay)
        self.features.append(never_worked)
        self.features.append(unknown_workclass)
        # fnlwgt (int)
        self.features.append(fnlwgt)
        # education (bool, one hot encoding)
        self.features.append(bachelors)
        self.features.append(some_college)
        self.features.append(_11th)
        self.features.append(hs_grad)
        self.features.append(prof_school)
        self.features.append(assoc_acdm)
        self.features.append(assoc_voc)
        self.features.append(_9th)
        self.features.append(_7th_8th)
        self.features.append(_12th)
        self.features.append(masters)
        self.features.append(_1st_4th)
        self.features.append(_10th)
        self.features.append(doctorate)
        self.features.append(_5th_6th)
        self.features.append(preschool)
        self.features.append(unknown_education)
        # education-num (int)
        self.features.append(education_num)
        # marital-status (bool, one hot encoding)
        self.features.append(married_civ_spouse)
        self.features.append(divorced)
        self.features.append(never_married)
        self.features.append(separated)
        self.features.append(widowed)
        self.features.append(married_spouse_absent)
        self.features.append(married_af_spouse)
        self.features.append(unknown_marital_status)
        # occupation (bool, one hot encoding)
        self.features.append(tech_support)
        self.features.append(craft_repair)
        self.features.append(other_service)
        self.features.append(sales)
        self.features.append(exec_managerial)
        self.features.append(prof_specialty)
        self.features.append(handlers_cleaners)
        self.features.append(machine_op_inspct)
        self.features.append(adm_clerical)
        self.features.append(farming_fishing)
        self.features.append(transport_moving)
        self.features.append(priv_house_serv)
        self.features.append(protective_serv)
        self.features.append(armed_forces)
        self.features.append(unknown_occupation)
        # relationship (bool, one hot encoding)
        self.features.append(wife)
        self.features.append(own_child)
        self.features.append(husband)
        self.features.append(not_in_family)
        self.features.append(other_relative)
        self.features.append(unmarried)
        self.features.append(unknown_relationship)
        # race (bool, one hot encoding)
        self.features.append(white)
        self.features.append(asian_pac_islander)
        self.features.append(amer_indian_eskimo)
        self.features.append(other)
        self.features.append(black)
        self.features.append(unknown_race)
        # sex (bool, 0=female 1=male)
        self.features.append(sex)
        # capital-gain (int)
        self.features.append(capital_gain)
        # capital-loss (int)
        self.features.append(capital_loss)
        # hours-per-week (int)
        self.features.append(hours_per_week)
        # native-country (bool, one hot encoding)
        self.features.append(united_states)
        self.features.append(cambodia)
        self.features.append(england)
        self.features.append(puerto_rico)
        self.features.append(canada)
        self.features.append(germany)
        self.features.append(outlying_us)
        self.features.append(india)
        self.features.append(japan)
        self.features.append(greece)
        self.features.append(south)
        self.features.append(china)
        self.features.append(cuba)
        self.features.append(iran)
        self.features.append(honduras)
        self.features.append(philippines)
        self.features.append(italy)
        self.features.append(poland)
        self.features.append(jamaica)
        self.features.append(vietnam)
        self.features.append(mexico)
        self.features.append(portugal)
        self.features.append(ireland)
        self.features.append(france)
        self.features.append(dominican_republic)
        self.features.append(laos)
        self.features.append(ecuador)
        self.features.append(taiwan)
        self.features.append(haiti)
        self.features.append(columbia)
        self.features.append(hungary)
        self.features.append(guatemala)
        self.features.append(nicaragua)
        self.features.append(scotland)
        self.features.append(thailand)
        self.features.append(yugoslavia)
        self.features.append(el_salvador)
        self.features.append(trinadad_tobago)
        self.features.append(peru)
        self.features.append(hong)
        self.features.append(holand_netherlands)
        self.features.append(unknown_country)
        # label (bool, 0=<50k, 1=>50k)
        self.label = label
        # distance from test datum, we will leave this blank for now
        self.distance_from_person = 0
        self.test_data_sorted_by_distance = []
        # we will need these specific values for imputation, and I honestly don't feel like looking through the features list to find them
        self.fnlwgt = fnlwgt
        self.education_num = education_num
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week

def get_data_from_csv(source):
    data = []
    with open(source) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            age = int(row[0])
            
            workclass = row[1]
            private = 1 if "Private" in workclass else 0
            self_emp_not_inc = 1 if "Self-emp-not-inc" in workclass else 0
            self_emp_inc = 1 if "Self-emp-inc" in workclass else 0 
            federal_gov = 1 if "Federal-gov" in workclass else 0
            local_gov = 1 if "Local-gov" in workclass else 0
            state_gov = 1 if "State-gov" in workclass else 0
            without_pay = 1 if "Without-pay" in workclass else 0
            never_worked = 1 if "Never-worked" in workclass else 0
            unknown_workclass = 1 if "?" in workclass else 0
            
            fnlwgt = int(row[2])

            education = row[3]
            bachelors = 1 if "Bachelors" in education else 0
            some_college = 1 if "Some-college" in education else 0
            _11th = 1 if "11th" in education else 0
            hs_grad = 1 if "HS-grad" in education else 0
            prof_school = 1 if "Prof-school" in education else 0
            assoc_acdm = 1 if "Assoc-acdm" in education else 0
            assoc_voc = 1 if "Assoc-voc" in education else 0
            _9th = 1 if "9th" in education else 0
            _7th_8th = 1 if "7th-8th" in education else 0
            _12th = 1 if "12th" in education else 0
            masters = 1 if "Masters" in education else 0
            _1st_4th = 1 if "1st-4th" in education else 0
            _10th = 1 if "10th" in education else 0
            doctorate = 1 if "Doctorate" in education else 0
            _5th_6th = 1 if "5th-6th" in education else 0
            preschool = 1 if "Preschool" in education else 0
            unknown_education = 1 if "?" in education else 0
            
            education_num = int(row[4]) if row[4]!="?" else -1
            
            marital_status = row[5]
            married_civ_spouse = 1 if "Married-civ-spouse" in marital_status else 0
            divorced = 1 if "Divorced" in marital_status else 0
            never_married = 1 if "Never-married" in marital_status else 0
            separated = 1 if "Separated" in marital_status else 0
            widowed = 1 if "Widowed" in marital_status else 0
            married_spouse_absent = 1 if "Married-spouse-absent" in marital_status else 0
            married_af_spouse = 1 if "Married-AF-spouse" in marital_status else 0
            unknown_marital_status = 1 if "?" in marital_status else 0
            
            occupation = row[6]
            tech_support = 1 if "Tech-support" in occupation else 0
            craft_repair = 1 if "Craft-repair" in occupation else 0
            other_service = 1 if "Other-service" in occupation else 0
            sales = 1 if "Sales" in occupation else 0
            exec_managerial = 1 if "Exec-managerial" in occupation else 0
            prof_specialty = 1 if "Prof-specialty" in occupation else 0
            handlers_cleaners = 1 if "Handlers-cleaners" in occupation else 0
            machine_op_inspct = 1 if "Machine-op-inspct" in occupation else 0
            adm_clerical = 1 if "Adm-clerical" in occupation else 0
            farming_fishing = 1 if "Farming-fishing" in occupation else 0
            transport_moving = 1 if "Transport-moving" in occupation else 0
            priv_house_serv = 1 if "Priv-house-serv" in occupation else 0
            protective_serv = 1 if "Protective-serv" in occupation else 0
            armed_forces = 1 if "Armed-Forces" in occupation else 0
            unknown_occupation = 1 if "?" in occupation else 0
            
            relationship = row[7]
            wife = 1 if "Wife" in relationship else 0
            own_child = 1 if "Own-child" in relationship else 0
            husband = 1 if "Husband" in relationship else 0
            not_in_family = 1 if "Not-in-family" in relationship else 0
            other_relative = 1 if "Other-relative" in relationship  else 0
            unmarried = 1 if "Unmarried" in relationship else 0
            unknown_relationship = 1 if "?" in relationship else 0
            
            race = row[8]
            white = 1 if "White" in race else 0
            asian_pac_islander = 1 if "Asian-Pac-Islander" in race else 0
            amer_indian_eskimo = 1 if "Amer-Indian-Eskimo" in race else 0
            other = 1 if "Other" in race else 0
            black = 1 if "Black" in race else 0
            unknown_race = 1 if race=="?" else 0
            
            sex = 1 if row[9]=="Male" else 0
            capital_gain = int(row[10]) if row[10]!="?" else -1
            capital_loss = int(row[11]) if row[11]!="?" else -1
            hours_per_week = int(row[12]) if row[12]!="?" else -1
            
            country = row[13]
            united_states = 1 if "United-States" in country else 0
            cambodia = 1 if "Cambodia" in country else 0
            england = 1 if "England" in country else 0
            puerto_rico = 1 if "Puerto-Rico" in country else 0
            canada = 1 if "Canada" in country else 0
            germany = 1 if "Germany" in country else 0
            outlying_us = 1 if "Outlying US" in country else 0
            india = 1 if "India" in country else 0
            japan = 1 if "Japan" in country else 0
            greece = 1 if "Greece" in country else 0
            south = 1 if "South" in country else 0
            china = 1 if "China" in country else 0
            cuba = 1 if "Cuba" in country else 0
            iran = 1 if "Iran" in country else 0
            honduras = 1 if "Honduras" in country else 0
            philippines = 1 if "Philippines" in country else 0
            italy = 1 if "Italy" in country else 0
            poland = 1 if "Poland" in country else 0
            jamaica = 1 if "Jamaica" in country else 0
            vietnam = 1 if "Vietnam" in country else 0
            mexico = 1 if "Mexico" in country else 0
            portugal = 1 if "Portugal" in country else 0
            ireland = 1 if "Ireland" in country else 0
            france = 1 if "France" in country else 0
            dominican_republic = 1 if "Dominican-Republic" in country else 0
            laos = 1 if "Laos" in country else 0
            ecuador = 1 if "Ecuador" in country else 0
            taiwan = 1 if "Taiwan" in country else 0
            haiti = 1 if "Haiti" in country else 0
            columbia = 1 if "Columbia" in country else 0
            hungary = 1 if "Hungary" in country else 0
            guatemala = 1 if "Guatemala" in country else 0
            nicaragua = 1 if "Nicaragua" in country else 0
            scotland = 1 if "Scotland" in country else 0
            thailand = 1 if "Thailand" in country else 0
            yugoslavia = 1 if "Yugoslavia" in country else 0
            el_salvador = 1 if "El-Salvador" in country else 0
            trinadad_tobago = 1 if "Trinadad" in country else 0
            peru = 1 if "Peru" in country else 0
            hong = 1 if "Hong" in country else 0
            holand_netherlands = 1 if "Holand" in country else 0
            unknown_country = 1 if "?" in country else 0

            label = 1 if ">50K" in row[14] else 0

            new_person = Person(age, private, self_emp_not_inc, self_emp_inc, federal_gov, local_gov, state_gov, without_pay, never_worked, unknown_workclass, fnlwgt, bachelors, some_college, _11th, hs_grad, prof_school, assoc_acdm, assoc_voc, _9th, _7th_8th, _12th, masters, _1st_4th, _10th, doctorate, _5th_6th, preschool, unknown_education, education_num, married_civ_spouse, divorced, never_married, separated, widowed, married_spouse_absent, married_af_spouse, unknown_marital_status, tech_support, craft_repair, other_service, sales, exec_managerial, prof_specialty, handlers_cleaners, machine_op_inspct, adm_clerical, farming_fishing, transport_moving, priv_house_serv, protective_serv, armed_forces, unknown_occupation, wife, own_child, husband, not_in_family, other_relative, unmarried, unknown_relationship, white, asian_pac_islander, amer_indian_eskimo, other, black, unknown_race, sex, capital_gain, capital_loss, hours_per_week, united_states, cambodia, england, puerto_rico, canada, germany, outlying_us, india, japan, greece, south, china, cuba, iran, honduras, philippines, italy, poland, jamaica, vietnam, mexico, portugal, ireland, france, dominican_republic, laos, ecuador, taiwan, haiti, columbia, hungary, guatemala, nicaragua, scotland, thailand, yugoslavia, el_salvador, trinadad_tobago, peru, hong, holand_netherlands, unknown_country, label)
            data.append(new_person)
    return data

def imputation(training_data, data_to_impute):
    avg_fnlwgt, avg_education_num, avg_capital_gain, avg_capital_loss, avg_hours_per_week = 0.0, 0.0, 0.0, 0.0, 0.0
    count_fnlwgt, count_education_num, count_capital_gain, count_capital_loss, count_hours_per_week = 0.0, 0.0, 0.0, 0.0, 0.0
    for person in training_data:
        if person.fnlwgt != -1:
            avg_fnlwgt += person.fnlwgt
            count_fnlwgt += 1
        if person.education_num != -1:
            avg_education_num += person.education_num
            count_education_num += 1
        if person.capital_gain != -1:
            avg_capital_gain += person.capital_gain
            count_capital_gain += 1
        if person.capital_loss != -1:
            avg_capital_loss += person.capital_loss
            count_capital_loss += 1
        if person.hours_per_week != -1:
            avg_hours_per_week += person.hours_per_week
            count_hours_per_week += 1
    avg_education_num /= count_education_num
    avg_education_num /= count_education_num
    avg_capital_gain /= count_capital_gain
    avg_capital_loss /= count_capital_loss
    avg_hours_per_week /= count_hours_per_week
    for person in data_to_impute:
        if person.fnlwgt == -1:
            person.fnlwgt = avg_fnlwgt
        if person.education_num == -1:
            person.education_num = avg_education_num
        if person.capital_gain == -1:
            person.capital_gain = avg_capital_gain
        if person.capital_loss == -1:
            person.capital_loss = avg_capital_loss
        if person.hours_per_week == -1:
            person.hours_per_week = avg_hours_per_week
    return data_to_impute

def get_averages(training_data):
    averages = []
    features_count = len(training_data[0].features)
    for i in range(features_count):
        sum = 0.0
        for j in range(len(training_data)):
            sum += training_data[j].features[i]
        averages.append(sum / len(training_data))
    return averages

def get_std_devs(training_data, averages):
    std_devs = []
    features_count = len(training_data[0].features)
    for i in range(features_count):
        sum = 0.0
        n = len(training_data)
        for j in range(n):
            sum += training_data[j].features[i] * training_data[j].features[i]
        variance = (sum/n) - (averages[i]*averages[i])
        std_devs.append(math.sqrt(variance))
    return std_devs

def z_score_normalization(data, averages, std_devs):
    features_count = len(data[0].features)
    for i in range(len(data)):
        for j in range(features_count):
            data[i].features[j] = ((data[i].features[j] - averages[j]) / std_devs[j]) if std_devs[j]!=0 else data[i].features[j]
    return data

def get_distance(x, y):
    features_count = len(x.features)
    sum = 0.0
    for i in range(features_count):
        sum += (x.features[i] - y.features[i])*(x.features[i] - y.features[i])
    return math.sqrt(sum)

def sort_training_data_by_distance(training_data, x):
    for y in training_data:
        y.distance_from_person = get_distance(x,y)
    training_data.sort(key=lambda x: x.distance_from_person, reverse=True)
    return training_data

def predict_label(x, k, training_data):
    yes, no = 0, 0
    if len(x.test_data_sorted_by_distance) == 0:
        x.test_data_sorted_by_distance = sort_training_data_by_distance(training_data, x)
    for y in x.test_data_sorted_by_distance[0:k]:
        if y.label == 1:
            yes += 1
        else:
            no += 1
    return 1 if yes > no else 0

def accuracy(k, test_data, training_data):
    correct, total = 0.0, 0.0
    for x in test_data:
        # print("Processing #" + str(total))
        if x.label == predict_label(x, k, training_data):
            correct += 1
        total += 1
    return correct/total

def main():
    print("Getting training data...\n")
    training_data = get_data_from_csv("census-income.data.csv")
    print("Getting test data...\n")
    test_data = get_data_from_csv("census-income.test.csv")
    print("Imputating...\n")
    training_data = imputation(training_data, training_data)
    test_data = imputation(training_data, test_data)
    # print("Getting averages...\n")
    # averages = get_averages(training_data)
    # print("Getting standard deviations...\n")
    # std_devs = get_std_devs(training_data, averages)
    # print("Z-score normalizing...\n")
    # training_data = z_score_normalization(training_data, averages, std_devs)
    # test_data = z_score_normalization(test_data, averages, std_devs)
    k_values = [1, 5, 9, 11, 21, 51, 101, 201, 401]
    print("Testing accuracy...")
    for k in k_values:
        print("k=" + str(k) + " ==> accuracy of " + str(accuracy(k, test_data[0:100], training_data)))
    print("DONE")

main()