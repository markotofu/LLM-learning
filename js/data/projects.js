/* ML Atlas – case studies and sample projects, attached to lessons by id in MLA.content.init (js/data/branches.js).
 *   Case study: { title, when, story, lesson, source: { title, url? } }        story and lesson are markdown
 *   Project:    { title, level: 'starter' | 'intermediate' | 'advanced', hours, goal, data, steps: [md], done, stretch: [md], skills: [text],
 *                 practice?: a bundled synthetic dataset (js/core/datasets.js) to start with in the browser }
 * Case studies are real, documented events: every number here comes from the cited source. Projects are meant for the
 * learner's own machine (most need pip packages or dataset downloads that the in-browser Python cannot reach).
 * Lessons list them after the key takeaways; the Projects page (#/projects) lists them all. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  const C = (title, when, story, lesson, source) => ({ title, when, story, lesson, source });
  const P = (o) => o;

  // ------------------------------------------------------------------ case studies
  MLA.data.cases = {
    'statistics': [C('Simpson\'s paradox in graduate admissions', 'Berkeley, 1973',
      `In the autumn of 1973 the University of California, Berkeley admitted about 44% of the 8,442 men and 35% of the 4,321 women who applied to its graduate programs. On those numbers alone it looked like bias against women.

Bickel, Hammel and O'Connell then looked department by department. In most departments women were admitted at a rate similar to men, and in several at a higher rate. The overall gap came from where people applied: women applied more often to departments that admitted a small share of their applicants, and men more often to departments that admitted a large share.`,
      `An overall rate can point one way while every subgroup points the other way, when a third variable (here, the department) affects both who is in each group and the outcome. Before drawing a conclusion from a comparison of rates, ask what else differs between the groups, and compare within strata.`,
      { title: 'Bickel, Hammel and O\'Connell, "Sex Bias in Graduate Admissions: Data from Berkeley", Science 187 (1975)', url: 'https://en.wikipedia.org/wiki/Simpson%27s_paradox' })],

    'stat-inference': [C('A headline change worth 12% of revenue', 'Bing, 2012',
      `An engineer working on Microsoft's Bing search engine suggested making ad headlines longer by moving text from the line below into the title. The idea sat in the backlog for months because nobody rated it highly. When it was finally tested as an online controlled experiment (an A/B test), revenue rose by 12%, worth more than $100 million a year in the US alone, with no harm to the user-experience metrics the team monitored.

The effect was so large that it set off an alert designed to catch bugs. The team checked the experiment and repeated it, and the result held.`,
      `Intuition is a poor guide to which changes matter; a well-run experiment with enough traffic and guardrail metrics is a better one. A surprisingly large effect deserves suspicion first: check the setup, look for a bug, and replicate before you ship.`,
      { title: 'Kohavi and Thomke, "The Surprising Power of Online Experiments", Harvard Business Review, September 2017', url: 'https://hbr.org/2017/09/the-surprising-power-of-online-experiments' })],

    'data-visualization': [C('The Challenger launch decision', '1986',
      `On the night before the space shuttle Challenger was launched on 28 January 1986, engineers at Morton Thiokol, the company that built its solid rocket boosters, recommended against launching in the forecast cold. Rubber O-rings sealed the joints of the boosters, several had been damaged on earlier flights, and the engineers believed that cold rubber sealed less well. Their charts did not persuade the managers, and the launch went ahead at an air temperature of about 36 °F (2 °C), far colder than any earlier launch; the coldest before had been 53 °F (12 °C). A joint failed, and the shuttle broke apart 73 seconds after lift-off. All seven crew members died.

The presidential commission that investigated the accident pointed to a flaw in how the evidence had been examined: the discussion before the launch looked only at the flights on which O-rings had been damaged, and in that subset temperature showed no clear pattern. Adding the flights without damage, which had all launched in warmer weather, made the relationship between cold and damage plain.`,
      `Which data you plot decides what you can see. Include the cases where nothing happened, put the variable the decision depends on (here, temperature) on an axis, and make the comparison that matters as direct as possible.`,
      { title: 'Report of the Presidential Commission on the Space Shuttle Challenger Accident (1986)', url: 'https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster' })],

    'cross-validation': [C('Hundreds of COVID-19 imaging models, none ready for the clinic', '2020 to 2021',
      `In 2020 many research groups published machine learning models that diagnosed COVID-19 or predicted its course from chest X-rays and CT scans, often with very high reported accuracy. Roberts and colleagues screened more than 2,000 such papers and reviewed 62 in detail. They found none of potential clinical use, because of methodological flaws or underlying biases.

Common problems: public datasets were combined so that the same images could end up in both training and test sets; some control images (healthy or pneumonia cases) came from children while the COVID-19 images came from adults, so a model could learn age instead of disease; and models were tested on data from the same sources they were trained on.`,
      `A validation score is only as honest as the split behind it. Split by patient, not by image; remove duplicates across sources; make sure positive and negative examples come from comparable populations and machines; and test on data from a different hospital before you trust the number.`,
      { title: 'Roberts et al., "Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans", Nature Machine Intelligence 3 (2021)', url: 'https://www.nature.com/articles/s42256-021-00307-0' })],

    'evaluation-metrics': [C('A sepsis alert checked outside its home', '2021',
      `The Epic Sepsis Model is a proprietary model, used in hundreds of US hospitals, that scores patients for their risk of sepsis. Wong and colleagues evaluated it at Michigan Medicine on 38,455 hospitalizations. The area under the ROC curve was 0.63, far below the figures reported by the developer.

At the alert threshold used in practice, the model missed 67% of the patients who developed sepsis, and it raised alerts for 18% of all hospitalized patients, so most alerts were false alarms that staff had to handle.`,
      `One headline metric is not enough, and a developer's own validation is an optimistic upper bound. Evaluate at the threshold you will actually use (sensitivity, precision and alert rate), and validate on data from a different site and time before relying on a model.`,
      { title: 'Wong et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients", JAMA Internal Medicine 181 (2021)' })],

    'overfitting': [C('Google Flu Trends', '2008 to 2015',
      `Google Flu Trends estimated flu activity in the US from the volume of 45 search queries, picked from the 50 million most common queries by how well they tracked official data from the Centers for Disease Control and Prevention (CDC). Published in Nature in 2009, it gave estimates about two weeks before the official reports.

Then it drifted. In the 2012 to 2013 season it predicted more than double the share of doctor visits for flu-like illness that the CDC reported. Lazer and colleagues pointed to overfitting (screening millions of queries against a few hundred data points finds terms that only happen to follow the winter season) and to changes in Google's own search product, which changed what people typed. Google stopped publishing the estimates in 2015.`,
      `Choosing among millions of candidate features against a short history fits noise. A model also goes stale when the process that produces its inputs changes. Keep a holdout that is later in time, prefer fewer and more plausible features, and keep measuring after launch.`,
      { title: 'Lazer, Kennedy, King and Vespignani, "The Parable of Google Flu: Traps in Big Data Analysis", Science 343 (2014)', url: 'https://en.wikipedia.org/wiki/Google_Flu_Trends' })],

    'naive-bayes': [C('A plan for spam', '2002',
      `In August 2002 the programmer Paul Graham published an essay describing a spam filter built on word probabilities. From his own collections of spam and legitimate mail, he estimated for each token the probability that a message containing it is spam. A new message was scored by combining the probabilities of its 15 most telling tokens with Bayes' rule.

He counted tokens in legitimate mail twice, which biased the filter against false positives, because losing a real message costs far more than letting one spam through. He reported that the filter caught about 99.5% of his spam with very few false positives. Filters of this kind were soon built into mail software.`,
      `A simple probabilistic model trained on your own data can work very well. The naive independence assumption is wrong, but the ranking it produces is often good enough; what matters more is careful handling of rare words and of the unequal costs of the two kinds of error.`,
      { title: 'Paul Graham, "A Plan for Spam" (2002)', url: 'http://www.paulgraham.com/spam.html' })],

    'ensembles': [C('Gradient boosting wins on tables', 'Kaggle, 2015',
      `The authors of XGBoost counted the winning solutions published on Kaggle's blog during 2015: 17 of the 29 used XGBoost. Eight of them used XGBoost alone, and most of the others combined it with neural networks in an ensemble. In the KDD Cup 2015 competition, every team in the top 10 used it.

The pattern has held up since: on structured, tabular data, boosted trees are usually the model to beat, while neural networks lead on images, audio and text.`,
      `For tabular problems, start with a well-tuned gradient-boosted tree model as the baseline. Blending different model types can add a little more, which matters in a competition and rarely justifies its cost in production.`,
      { title: 'Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016', url: 'https://arxiv.org/abs/1603.02754' })],

    'recommender-systems': [C('The Netflix Prize', '2006 to 2009',
      `In October 2006 Netflix offered $1 million to the first team that beat its own rating predictor, Cinematch, by 10% in RMSE on a hidden test set. The training data held about 100 million ratings from 480,000 users on 17,770 movies.

The prize was won in September 2009 by the team BellKor's Pragmatic Chaos, with a 10.06% improvement, submitted 20 minutes before a rival team with the same score. The ideas that mattered were matrix factorization (learned vectors for users and movies, plus biases and effects of time) and blending hundreds of models.

Netflix later wrote that it had put two earlier components into production, a factorization model and restricted Boltzmann machines, but not the full winning blend: the extra accuracy did not justify the engineering effort, and its business had shifted to streaming, where what people choose to watch matters more than star ratings.`,
      `A leaderboard metric is a proxy for what the product needs. Matrix factorization was the lasting idea; the last fraction of a percent from a huge blend was not worth running. Pick a metric that tracks the real goal, and weigh accuracy against cost.`,
      { title: 'Netflix Prize (Wikipedia), and the Netflix Technology Blog, "Netflix Recommendations: Beyond the 5 stars" (2012)', url: 'https://en.wikipedia.org/wiki/Netflix_Prize' })],

    'cnn': [C('AlexNet and ImageNet', '2012',
      `The 2012 ImageNet Large Scale Visual Recognition Challenge asked entrants to classify images into 1,000 classes, with about 1.2 million training images. A convolutional network by Krizhevsky, Sutskever and Hinton reached a top-5 test error of 15.3%. The second-best entry, built on hand-designed image features, had 26.2%.

The network had about 60 million parameters, five convolutional layers and three fully connected ones, and was trained for five to six days on two consumer GPUs. It used ReLU activations, dropout, data augmentation and overlapping max pooling. Within a few years almost every entry in the challenge was a deep convolutional network.`,
      `Learned features beat hand-designed ones once there is enough labelled data and compute to train them. The ingredients used here (ReLU, dropout, augmentation, training on GPUs) are still standard practice.`,
      { title: 'Krizhevsky, Sutskever and Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NeurIPS 2012', url: 'https://en.wikipedia.org/wiki/AlexNet' })],

    'seq2seq': [C('Neural machine translation at Google', '2016',
      `In September 2016 Google described GNMT, its neural machine translation system: an encoder and a decoder, each a stack of 8 LSTM layers with residual connections, joined by attention. Words were split into subword units (wordpieces), which handled rare words and names.

In side-by-side ratings by human evaluators on several language pairs, GNMT reduced translation errors by about 60% on average compared with Google's phrase-based production system, which had been built from many separately engineered parts. It replaced that system for Chinese to English the same month.`,
      `A single sequence-to-sequence model with attention, trained end to end, could replace a complex hand-built pipeline, and subword units solved the rare-word problem. A year later the transformer replaced the recurrent layers.`,
      { title: 'Wu et al., "Google\'s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation" (2016)', url: 'https://arxiv.org/abs/1609.08144' })],

    'pretraining': [C('Chinchilla: a smaller model, trained longer', '2022',
      `DeepMind trained over 400 language models, from 70 million to over 16 billion parameters on 5 to 500 billion tokens, to answer one question: for a fixed compute budget, how should you split it between model size and training data? Their answer was to grow both in equal proportion, which works out to roughly 20 training tokens per parameter.

To test it, they trained Chinchilla: 70 billion parameters on 1.4 trillion tokens, with the same compute as their earlier 280 billion parameter model Gopher, which had seen 300 billion tokens. Chinchilla outperformed Gopher, GPT-3 (175 billion parameters) and Megatron-Turing NLG (530 billion) on a wide range of evaluations, and it is cheaper to run because it is smaller.`,
      `Many large models of the time were undertrained for their size. When a model will answer many requests, a smaller model trained on more data also saves money at inference, which is why later open models train on far more than 20 tokens per parameter.`,
      { title: 'Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)', url: 'https://arxiv.org/abs/2203.15556' })],

    'alignment': [C('InstructGPT: a 1.3 billion parameter model preferred to a 175 billion one', '2022',
      `OpenAI fine-tuned GPT-3 to follow instructions in three steps: supervised fine-tuning on demonstrations written by human labelers; training a reward model on labelers' rankings of several outputs for the same prompt; and optimizing the language model against that reward model with PPO (reinforcement learning from human feedback).

On prompts that customers had submitted to the API, labelers preferred the outputs of the 1.3 billion parameter InstructGPT model to those of the 175 billion parameter GPT-3, a model more than 100 times larger. InstructGPT also made up facts less often and produced less toxic text, though it still made simple mistakes.`,
      `Predicting the next token well is not the same as doing what users want. A modest amount of human preference data moved quality more than a 100-fold increase in size. The three-step recipe (supervised fine-tuning, a reward model, reinforcement learning) became the template for chat assistants.`,
      { title: 'Ouyang et al., "Training language models to follow instructions with human feedback" (2022)', url: 'https://arxiv.org/abs/2203.02155' })],

    'reinforcement-learning': [C('AlphaGo and move 37', '2016',
      `In March 2016 DeepMind's AlphaGo beat Lee Sedol, one of the strongest Go players in the world, by four games to one in Seoul. It combined deep networks with Monte Carlo tree search. A policy network, trained first on games by strong human players and then improved by playing against itself, proposed moves; a value network, trained on positions from self-play, estimated who was winning; the search combined the two.

In game 2 AlphaGo played move 37, a move that commentators first took for a mistake; its policy network had estimated that a human would play it with a probability of about 1 in 10,000. In game 4 Lee Sedol won with an unexpected move of his own, move 78.`,
      `Search combined with learned evaluation can go beyond the data it started from, and self-play produces its own training signal. A later version, AlphaGo Zero, learned from self-play alone, without human games.`,
      { title: 'Silver et al., "Mastering the game of Go with deep neural networks and tree search", Nature 529 (2016)', url: 'https://en.wikipedia.org/wiki/AlphaGo_versus_Lee_Sedol' })],

    'ai-for-science': [C('AlphaFold 2 at CASP14', '2020',
      `CASP is a blind test held every two years: groups predict the 3D structures of proteins whose structures have been measured in the lab but not yet published. At CASP14 in 2020, DeepMind's AlphaFold 2 reached a median GDT score of 92.4 across all targets. GDT runs from 0 to 100, and a score of about 90 is considered competitive with experimental methods.

The model used attention over alignments of related protein sequences and over pairs of residues, and predicted the structure end to end. In 2021 DeepMind and EMBL's European Bioinformatics Institute opened a database of predicted structures, which grew to more than 200 million proteins. Demis Hassabis and John Jumper shared the 2024 Nobel Prize in Chemistry for this work, together with David Baker for computational protein design.`,
      `A well-defined problem with a blind benchmark, decades of carefully curated data (the Protein Data Bank) and an architecture built around the structure of the problem turned a 50-year-old challenge into a tool that biologists use every day.`,
      { title: 'Jumper et al., "Highly accurate protein structure prediction with AlphaFold", Nature 596 (2021)', url: 'https://en.wikipedia.org/wiki/AlphaFold' })],

    'responsible-ai': [
      C('Gender Shades', '2018',
        `Joy Buolamwini and Timnit Gebru evaluated three commercial face analysis services that classify gender (from IBM, Microsoft and Face++) on a new benchmark of 1,270 photos of members of parliament from three African and three European countries, balanced by gender and skin type.

The error rate was at most 0.8% for lighter-skinned men and up to 34.7% for darker-skinned women. The overall accuracy of each service looked high, because the groups with the most errors were a small part of the usual benchmarks. After the results were shared, the companies released updated versions.`,
        `Measure performance for each subgroup, not only overall; a benchmark that is balanced across groups shows what an overall accuracy hides. Who is in the evaluation data decides what you can find out.`,
        { title: 'Buolamwini and Gebru, "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification", Proceedings of Machine Learning Research 81 (2018)', url: 'https://proceedings.mlr.press/v81/buolamwini18a.html' }),
      C('A hiring model that learned from the past', '2018',
        `Reuters reported in October 2018 that Amazon had dropped an experimental tool that rated job applicants. It was trained on resumes submitted to the company over 10 years, most of them from men, and it learned to penalize resumes that contained the word "women's" (as in "women's chess club captain") and to downgrade graduates of two all-women's colleges.

Engineers edited the model to be neutral to those terms, but could not be sure it would not find other ways to sort candidates that were just as unfair, and the project was abandoned. According to Reuters, recruiters looked at its rankings but never relied on them alone.`,
        `A model trained on past decisions learns to repeat them. Removing a sensitive word does not remove bias when other features act as proxies; audit outcomes by group, and question whether historical labels are the target you want.`,
        { title: 'Dastin, "Amazon scraps secret AI recruiting tool that showed bias against women", Reuters, 10 October 2018' })],

    'interpretability': [C('Asthma and pneumonia', 'reported in 2015',
      `In a 1990s project to predict which pneumonia patients were at low risk of dying, so that they could be treated at home, a rule-based model learned that patients with asthma had a lower risk. The pattern was real in the data: pneumonia patients with asthma were usually admitted straight to intensive care, and the extra care lowered their risk. A model used to decide who could go home would have sent home the patients who needed care the most.

The team noticed because the rules were readable. A neural network trained on the same data was more accurate, but they chose not to deploy it, because it could have learned the same pattern without anyone seeing it. Caruana and colleagues later fitted an intelligible additive model to similar data, found the asthma effect and several others like it, and showed how such effects can be inspected and corrected before deployment.`,
      `A model learns correlations in the data as it was produced, including the effects of treatments and policies. Look at what a model has learned (feature effects, rules) together with domain experts before you use it to make decisions.`,
      { title: 'Caruana et al., "Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission", KDD 2015' })],

    'causal-inference': [C('Hormone therapy: observational studies against a randomized trial', '2002',
      `In the 1980s and 1990s, large observational studies, such as the Nurses' Health Study, found that women who took hormone therapy after menopause had lower rates of coronary heart disease, and the therapy was widely prescribed partly for that reason.

The Women's Health Initiative tested estrogen plus progestin in a randomized trial of about 16,600 women. The trial was stopped early, in 2002: the treated group had more coronary heart disease, strokes, blood clots and breast cancer, not less. A leading explanation for the gap is confounding: women who chose hormone therapy were, on average, healthier and better off than those who did not. The age at which women started therapy also differed between the studies.`,
      `An association in observational data is not a causal effect when the choice of treatment is linked to other causes of the outcome. Adjust for the confounders you can measure, look for natural experiments, and for important decisions, run a randomized trial.`,
      { title: 'Writing Group for the Women\'s Health Initiative Investigators, "Risks and Benefits of Estrogen Plus Progestin in Healthy Postmenopausal Women", JAMA 288 (2002)', url: 'https://en.wikipedia.org/wiki/Women%27s_Health_Initiative' })],

    'rag': [C('Invented case law in a court filing', '2023',
      `In Mata v. Avianca, a personal injury case in a US federal court in New York, the plaintiff's lawyers filed a brief in 2023 that cited six court decisions that did not exist. One of the lawyers had used ChatGPT for legal research. It produced plausible case names, quotations and citations, and when he asked whether the cases were real, it said they were.

The judge fined two lawyers and their firm $5,000 and required them to notify the judges who had been named as authors of the invented opinions.`,
      `A language model without retrieval produces text that looks like a source; it is not a source. A grounded system retrieves documents, answers from them, cites them, and lets code or a person check that every citation exists and says what is claimed.`,
      { title: 'Mata v. Avianca, Inc., US District Court for the Southern District of New York (2023)', url: 'https://en.wikipedia.org/wiki/Mata_v._Avianca,_Inc.' })],

    'ai-engineering': [C('An airline held to its chatbot\'s answer', '2024',
      `In 2022 a customer asked the chatbot on Air Canada's website about bereavement fares after a relative died. The chatbot told him he could buy a regular ticket and apply for the reduced fare within 90 days. The airline's actual policy did not allow that after travel, and it refused the refund.

In February 2024 the Civil Resolution Tribunal of British Columbia decided for the customer. Air Canada had argued that the chatbot was responsible for its own statements; the tribunal rejected this, holding that the airline is responsible for all the information on its website, whether it comes from a static page or a chatbot. It ordered the airline to pay the fare difference with interest and fees, a little over 800 Canadian dollars.`,
      `What an LLM feature tells users is your organization's statement. Ground answers in the actual policy documents, test the questions that involve money or legal commitments, show the source, and route such questions to a fixed answer or a person.`,
      { title: 'Moffatt v. Air Canada, 2024 BCCRT 149' })],

    'gnn': [C('Graph networks in Google Maps arrival times', '2020',
      `DeepMind and Google Maps split road networks into "supersegments" (sequences of adjacent road segments that share significant traffic) and used a graph neural network to predict travel times through them, combining live traffic with historical patterns. Messages passed between neighbouring segments let a prediction take into account the traffic ahead and around.

After deployment the model improved the accuracy of real-time arrival estimates by up to 50% in cities such as Berlin, Jakarta, São Paulo, Sydney, Tokyo and Washington, D.C. Training was unstable at first; the team adapted the learning rate during training (with MetaGradients) and averaged model weights over training steps to make it reliable.`,
      `When the data is a network, message passing lets each prediction use its neighbours. A production graph model needs the same unglamorous work as any other: stable training, evaluation in each city, and careful comparison with the system it replaces.`,
      { title: 'Derrow-Pinion et al., "ETA Prediction with Graph Neural Networks in Google Maps", CIKM 2021' })],

    'contextual-bandits': [C('Personalized news on the Yahoo front page', '2010',
      `Li, Chu, Langford and Schapire treated the choice of which story to feature in the Today Module of the Yahoo front page as a contextual bandit: user and article features are the context, and a click is the reward. Their algorithm, LinUCB, fits a ridge regression for each article and adds a bonus for uncertainty, so that articles with little data still get shown.

They evaluated it offline on more than 33 million logged events. This was possible without bias because the logging policy had chosen articles uniformly at random, so each logged choice could be replayed fairly. LinUCB gave a 12.5% lift in clicks over a standard bandit that ignored the context.`,
      `A bandit learns while it serves, balancing exploration and exploitation for each user. Logging part of the traffic with known random choices is what makes honest offline evaluation of new policies possible later.`,
      { title: 'Li, Chu, Langford and Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW 2010', url: 'https://arxiv.org/abs/1003.0146' })],

    'time-series': [C('The M4 forecasting competition', '2018',
      `The M4 competition asked participants to forecast 100,000 real time series, from yearly to hourly, drawn from business, finance, economics, demography and other fields. Of the 17 most accurate methods, 12 were combinations of mostly statistical approaches.

The winner, by Slawek Smyl of Uber, was a hybrid: exponential smoothing for each series' level and seasonality, inside a recurrent neural network trained across all series. The six pure machine learning methods that were submitted all did poorly: none beat a simple combination of statistical benchmarks, and only one beat a seasonal naive forecast.`,
      `In forecasting, simple baselines (naive, seasonal naive, exponential smoothing) and combinations of methods are hard to beat, and learning across many related series helps. Always compare a new model with these baselines on a holdout that comes after the training period.`,
      { title: 'Makridakis, Spiliotis and Assimakopoulos, "The M4 Competition: Results, findings, conclusion and way forward", International Journal of Forecasting 34 (2018)', url: 'https://en.wikipedia.org/wiki/Makridakis_Competitions' })],

    'ml-security': [C('Stickers that turn a stop sign into a speed limit', '2018',
      `Eykholt and colleagues placed black and white stickers on a real stop sign, in a pattern computed by optimizing against a road sign classifier. The classifier read the sign as "Speed Limit 45" in all of the images taken in the lab and in 84.8% of the video frames recorded from a moving car. To a person, the stickers looked like graffiti.

The pattern was optimized over many angles, distances and lighting conditions, so that it kept working in the physical world rather than only on a single digital image.`,
      `A model's input is an attack surface. Test against adversarial inputs that match your threat model, combine the model with other signals (for example a map of where stop signs are), and do not let one classifier make safety-critical decisions on its own.`,
      { title: 'Eykholt et al., "Robust Physical-World Attacks on Deep Learning Visual Classification", CVPR 2018', url: 'https://arxiv.org/abs/1707.08945' })],

    'data-centric': [C('Label errors in famous test sets', '2021',
      `Northcutt, Athalye and Mueller used confident learning to find likely label errors in the test sets of 10 widely used datasets, then had people check the candidates. They estimated an average of at least 3.3% errors across the 10 test sets; about 6% of the ImageNet validation set was mislabeled.

The errors were large enough to change conclusions. On ImageNet with corrected labels, a ResNet-18 would outperform a ResNet-50 if the share of originally mislabeled test examples were just 6% higher.`,
      `Benchmark labels are data too, and noisy test labels can reorder models. Audit labels with tools that rank suspicious examples, fix the test set first, and expect a cleaner dataset to be worth more than a bigger model.`,
      { title: 'Northcutt, Athalye and Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", NeurIPS 2021 Datasets and Benchmarks', url: 'https://arxiv.org/abs/2103.14749' })],

    'calibration-uncertainty': [C('Zillow Offers', '2018 to 2021',
      `Zillow, known for its Zestimate home value estimates, started Zillow Offers in 2018 to buy homes directly, renovate them and sell them again, with purchase prices guided by its models. In 2021 it bought homes at a fast pace.

In November 2021 Zillow announced it would wind the business down. It wrote down about $304 million in the third quarter because it had bought many homes for more than it now expected to sell them for, and it planned to cut about 25% of its staff. Its chief executive said the unpredictability of home price forecasts was far greater than the company had expected.`,
      `Acting on point predictions at scale needs honest uncertainty. A forecast error of a few percent becomes a loss when the margin is a few percent, and the sellers who accept an offer are more likely to be the ones the model overpriced. Price in the uncertainty, check calibration as conditions change, and size decisions to what the model can actually predict.`,
      { title: 'Zillow Group, third quarter 2021 shareholder letter (November 2021)', url: 'https://en.wikipedia.org/wiki/Zillow' })],

    'tokenization': [C('SolidGoldMagikarp: tokens the model never learned', '2023',
      `In early 2023 Jessica Rumbelow and Matthew Watkins found that GPT-2 and GPT-3 models behaved strangely when given certain tokens, such as " SolidGoldMagikarp", which was a Reddit username. Asked to repeat the token, the models refused, changed the subject or produced unrelated words.

These strings had become single tokens because they were frequent in the text used to build the tokenizer. They were almost absent from the text the model itself was trained on, so their embeddings were barely trained and the model had no idea what they meant.`,
      `The tokenizer is part of the model. Build it from the same kind of data the model trains on, look for tokens that are rare in the training data, and test how the model handles unusual strings.`,
      { title: 'Rumbelow and Watkins, "SolidGoldMagikarp (plus, prompt generation)", LessWrong, February 2023' })],

    'llm-serving': [C('PagedAttention and vLLM', '2023',
      `Serving systems used to store each request's key-value cache in one contiguous block of memory, reserved for the maximum possible length. Much of that memory went unused: the vLLM authors measured that existing systems used only 20% to 38% of their KV cache memory for actual tokens.

PagedAttention stores the cache in small fixed-size blocks, like pages in an operating system, allocated as a sequence grows and shareable between sequences (for example when several samples share one prompt). With more requests fitting in memory, vLLM served 2 to 4 times more requests per second than FasterTransformer and Orca at the same latency.`,
      `LLM serving is limited by memory. How the KV cache is managed decides how many requests fit in a batch, and the batch size decides throughput and cost per token.`,
      { title: 'Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023', url: 'https://arxiv.org/abs/2309.06180' })],

    'mlops': [C('Uber\'s Michelangelo platform', '2017',
      `As machine learning spread across Uber, teams built models in their own ways, with no standard way to manage training data, compare experiments or deploy to production, and many models never shipped. Michelangelo, described publicly in 2017, gave every team the same workflow in six steps: manage data, train models, evaluate models, deploy models, make predictions, and monitor predictions.

A shared feature store let teams reuse curated features in both training and serving, so a feature was computed the same way offline and online, which removes a common source of silent errors.`,
      `Most of the work in production machine learning is the workflow around the model. Standardize data, features, training, deployment and monitoring once, so that features match between training and serving and every model is watched after launch.`,
      { title: 'Hermann and Del Balso, "Meet Michelangelo: Uber\'s Machine Learning Platform", Uber Engineering blog, September 2017' })],

    'speech-audio': [C('Whisper: 680,000 hours of imperfect transcripts', '2022',
      `OpenAI trained Whisper, a transformer encoder-decoder that reads log-mel spectrograms, on 680,000 hours of audio paired with transcripts collected from the internet: 438,000 hours of English, and the rest in 96 other languages or translated into English. Because many transcripts on the web are themselves machine-generated, the team filtered those out with heuristics.

Without being trained on any particular benchmark, Whisper was much more robust than models tuned to one. Compared with a model of equal accuracy on the LibriSpeech benchmark, it made about 55% fewer errors on average across other speech recognition datasets.`,
      `For robustness, a large and varied dataset, even with noisy labels, can matter more than clean data from one source. A model that tops one benchmark can be brittle elsewhere, so evaluate on many datasets.`,
      { title: 'Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (2022)', url: 'https://arxiv.org/abs/2212.04356' })]
  };

  // ------------------------------------------------------------------ sample projects
  MLA.data.projects = {
    'linear-regression': [P({ title: 'House prices from census data', level: 'starter', hours: 4,
      goal: `Predict the median house value of a California district from its census statistics, and explain which features drive the prediction.`,
      data: `The California housing data: 20,640 districts from the 1990 US census, with 8 numeric features (median income, house age, rooms, location and more) and the median house value in units of $100,000. Load it with \`sklearn.datasets.fetch_california_housing(as_frame=True)\`.`,
      steps: [
        `Split off a test set (20%) before looking closely at the data, with a fixed random seed.`,
        `Explore the training set: a histogram of each feature and of the target, and price against median income. Note the cap at 5.0: values above $500,000 were recorded as $500,000.`,
        `Fit a baseline that always predicts the training mean, then \`LinearRegression\` in a \`Pipeline\` with \`StandardScaler\`. Report RMSE and MAE with 5-fold cross-validation.`,
        `Read the coefficients of the scaled model: which features push the price up or down, and by how much per standard deviation?`,
        `Add features: rooms per household, bedrooms per room, people per household. Keep each one only if cross-validated error improves.`,
        `Plot residuals against predictions and on a map (longitude and latitude). Where is the model wrong in a systematic way?`,
        `At the very end, evaluate the chosen model once on the test set.`],
      done: `You can state the test RMSE in dollars, how much better it is than the mean baseline, and the three most important features with the direction of their effect.`,
      stretch: [`Try \`Ridge\` over a grid of alphas, and degree-2 polynomial features.`, `Compare with \`HistGradientBoostingRegressor\` and explain the gap.`, `Predict the logarithm of the price and compare the errors.`],
      skills: ['train and test splits', 'pipelines and scaling', 'RMSE and MAE', 'feature engineering', 'residual analysis'] })],

    'logistic-regression': [P({ title: 'An SMS spam filter', level: 'starter', hours: 4,
      goal: `Classify text messages as spam or not, choose a threshold that keeps false alarms rare, and see which words the model relies on.`,
      data: `The SMS Spam Collection from the UCI Machine Learning Repository: 5,574 English text messages, 747 of them spam (about 13%). The file is tab-separated with no header: a label, then the message.`,
      steps: [
        `Load the file with pandas, check the class balance, and split 80/20 with \`stratify\` so both sets keep the 13% spam rate.`,
        `Turn messages into features with \`TfidfVectorizer\` (try single words and word pairs), fitted on the training messages only.`,
        `Fit \`LogisticRegression\` and report precision, recall and F1 for the spam class, and the confusion matrix. Accuracy alone misleads here: predicting "not spam" every time scores 87%.`,
        `Plot the precision-recall curve and choose the threshold that keeps precision at 99% or above: a real message lost in the spam folder costs more than a spam that gets through.`,
        `List the 20 words with the largest positive and the 20 with the largest negative coefficients. Do they make sense?`,
        `Read every misclassified test message and group the errors by cause.`],
      done: `You can report spam recall at your chosen precision, explain the threshold, and show an example of each kind of error.`,
      stretch: [`Compare with multinomial naive Bayes on the same features.`, `Add character n-grams (\`analyzer="char_wb"\`) to catch disguised words such as "fr33".`, `Wrap the model in a function that takes a message and returns a label and a probability.`],
      skills: ['text features', 'class imbalance', 'precision and recall', 'choosing a threshold', 'error analysis'] })],

    'naive-bayes': [P({ title: 'Sort forum posts by topic', level: 'starter', hours: 3,
      goal: `Train a multinomial naive Bayes classifier that assigns newsgroup posts to one of 20 topics, and find out which topics it confuses.`,
      data: `The 20 Newsgroups dataset: about 18,800 posts, split by date into training and test sets. Load it with \`sklearn.datasets.fetch_20newsgroups\` and \`remove=("headers", "footers", "quotes")\`; the headers leak the answer.`,
      steps: [
        `Load the training and test sets with headers, footers and quotes removed, and read a few posts from different groups.`,
        `Vectorize with \`CountVectorizer\`, then with \`TfidfVectorizer\`, fitted on the training posts only.`,
        `Fit \`MultinomialNB\` and tune the smoothing parameter alpha with cross-validation over values from 0.001 to 1.`,
        `Report accuracy and macro-averaged F1 on the test set, and plot the 20 by 20 confusion matrix.`,
        `For the three most confused pairs of topics, print the words with the largest difference in log-probability between the two classes.`,
        `Compare with logistic regression on the same features: accuracy, training time and prediction time.`],
      done: `You can give the best alpha, test accuracy and macro F1, and explain the two most common confusions with the words behind them.`,
      stretch: [`Run again with the headers kept and measure how much the leak inflates accuracy.`, `Try \`ComplementNB\`, a variant designed for imbalanced text classes.`],
      skills: ['bag of words', 'Laplace smoothing', 'macro F1', 'confusion matrices', 'leakage checks'] })],

    'sklearn-workflow': [P({ title: 'Income prediction with one complete pipeline', level: 'intermediate', hours: 6,
      goal: `Build one scikit-learn pipeline that takes raw census rows (numbers, categories and missing values) to a tuned, calibrated prediction of whether a person's income is above $50,000.`,
      data: `The Adult (Census Income) dataset: 48,842 people from the 1994 US census, with 14 features such as age, education, occupation and hours worked per week. Load it with \`sklearn.datasets.fetch_openml("adult", version=2, as_frame=True)\`. Some columns have missing values.`,
      steps: [
        `Hold out 20% as a test set, stratified by the label (about 24% of people earn more than $50,000).`,
        `Write a \`ColumnTransformer\`: impute and scale the numeric columns; impute and one-hot encode the categorical ones with \`handle_unknown="ignore"\`.`,
        `Put it in a \`Pipeline\` with \`LogisticRegression\` and record the 5-fold cross-validated ROC AUC as the baseline.`,
        `Swap in \`HistGradientBoostingClassifier\` and tune two or three hyperparameters with \`RandomizedSearchCV\` over the whole pipeline, so the preprocessing is refit inside every fold.`,
        `Check calibration with a reliability curve; if needed, wrap the model in \`CalibratedClassifierCV\`.`,
        `Evaluate once on the test set, and compare error rates between groups (sex, race).`,
        `Save the fitted pipeline with \`joblib\`, load it again, and predict for one new person given as a dictionary.`],
      done: `A single saved pipeline that accepts raw rows, with its test ROC AUC, a calibration plot and a table of error rates by group.`,
      stretch: [`Add permutation importance and partial dependence plots for the top features.`, `Choose a decision threshold from a stated cost of false positives and false negatives.`],
      skills: ['ColumnTransformer', 'pipelines without leakage', 'hyperparameter search', 'calibration', 'saving models'] })],

    'ensembles': [P({ title: 'Predict which customers will leave', level: 'intermediate', hours: 6, practice: 'telco_churn_synthetic.csv',
      goal: `Predict churn for a telecom company with gradient-boosted trees, and turn the predictions into a retention plan with an expected value.`,
      data: `The IBM Telco Customer Churn sample (on Kaggle as "Telco Customer Churn"): 7,043 customers and 21 columns (contract type, tenure, monthly charges, services used); about 26% churned.`,
      steps: [
        `Clean the data: \`TotalCharges\` is stored as text, with blanks for brand-new customers. Convert it and decide what the blanks mean.`,
        `Split 80/20, stratified. Fit a logistic regression baseline and report ROC AUC and the precision among the 10% of customers with the highest scores.`,
        `Train a random forest and a gradient-boosted model (\`HistGradientBoostingClassifier\`, XGBoost or LightGBM), tuning the learning rate, depth and number of trees with early stopping on a validation split.`,
        `Compare the models on the same cross-validation folds. Is the gain over the baseline larger than the variation between folds?`,
        `Explain the model: permutation importance overall, and the main reasons behind a few individual high-risk customers.`,
        `Suppose a retention offer costs $50 and keeps a leaving customer worth $500 with probability 0.3. Compute the expected profit of contacting the top k customers for each k, and pick the best k.`],
      done: `A ranked list of customers to contact, the expected profit of the plan compared with contacting nobody or everybody, and the three main drivers of churn.`,
      stretch: [`Stack the tree model with the logistic regression and see whether the blend helps.`, `Use SHAP values to explain individual predictions.`],
      skills: ['gradient boosting', 'early stopping', 'model comparison', 'feature importance', 'decisions from costs'] })],

    'kmeans': [P({ title: 'Customer segments from purchase history', level: 'intermediate', hours: 5,
      goal: `Group the customers of an online shop by how recently, how often and how much they buy, and describe each group so a marketing team could act on it.`,
      data: `The Online Retail dataset (UCI): about 540,000 transactions from a UK online shop between December 2010 and December 2011, with invoice number, product, quantity, unit price, date and customer ID.`,
      steps: [
        `Clean: drop rows without a customer ID, set aside cancellations (invoice numbers that start with "C"), and remove zero or negative prices.`,
        `Build one row per customer: recency (days since the last purchase), frequency (number of invoices) and monetary value (total spend). These are the RFM features.`,
        `The features are very skewed: take logarithms, then standardize them.`,
        `Run k-means for k from 2 to 10. Plot inertia (the elbow) and the silhouette score, and choose k from both and from how usable the groups are.`,
        `Profile each cluster: its size and the median recency, frequency and spend. Give each one a plain name, such as "loyal big spenders" or "lapsed one-time buyers".`,
        `Check stability: rerun with different seeds and on a random 80% of the customers. Do the same groups come back?`],
      done: `A table of 3 to 6 named segments with their sizes and typical behaviour, and evidence that they are stable.`,
      stretch: [`Compare with a Gaussian mixture model and with hierarchical clustering.`, `Add the share of spending per product category and see whether new segments appear.`],
      skills: ['building features', 'scaling for distances', 'choosing k', 'profiling clusters', 'stability checks'] })],

    'pca': [P({ title: 'Eigenfaces: compress and recognize faces', level: 'intermediate', hours: 4,
      goal: `Use PCA to compress face images into a few dozen numbers, see what the components look like, and recognize people from the compressed version.`,
      data: `The Olivetti faces: 400 grayscale images of 40 people, 10 images each, 64 by 64 pixels. Load them with \`sklearn.datasets.fetch_olivetti_faces()\`.`,
      steps: [
        `Split so that each person has 7 images in training and 3 in test (stratify by person).`,
        `Fit PCA on the training images (4,096 pixels each) and plot the cumulative explained variance. How many components keep 90% and 95%?`,
        `Show the mean face and the first 16 components as images, and describe what the first few seem to capture (lighting, the shape of the face).`,
        `Rebuild a few test faces from 10, 50 and 150 components and show them next to the originals.`,
        `Classify the test faces with a 1-nearest-neighbour classifier on the PCA coordinates, and plot accuracy against the number of components.`],
      done: `A plot of accuracy against the number of components, reconstructions at three sizes, and a sentence on the trade-off between compression and accuracy.`,
      stretch: [`Replace the nearest-neighbour classifier with a linear SVM.`, `Compare PCA with random projections to the same number of dimensions.`],
      skills: ['PCA', 'explained variance', 'reconstruction', 'dimensionality reduction before classification'] })],

    'anomaly-detection': [P({ title: 'Catch card fraud among normal payments', level: 'intermediate', hours: 6, practice: 'card_transactions_synthetic.csv',
      goal: `Flag fraudulent card transactions when fraud is 0.17% of the data, and evaluate the detector with metrics that suit rare events.`,
      data: `The Credit Card Fraud Detection dataset (on Kaggle, from the Machine Learning Group of the Université Libre de Bruxelles): 284,807 European card transactions over two days in September 2013, 492 of them fraudulent. Features V1 to V28 are PCA components; Time and Amount are raw.`,
      steps: [
        `Split by time, not at random: train on the first 70% of the period and test on the rest, as a deployed detector would see it.`,
        `Unsupervised first: fit an \`IsolationForest\` on the training data and rank the test transactions by anomaly score.`,
        `Supervised next: a logistic regression and a gradient-boosted model, with and without class weights. Never oversample before splitting.`,
        `Evaluate with the precision-recall curve and average precision rather than accuracy or ROC AUC, and report recall at a fixed alert budget, such as the top 100 alerts per day.`,
        `Compare the unsupervised and supervised detectors at the same alert budget.`,
        `Look at the frauds the best model misses: what do their amounts and times look like?`],
      done: `Average precision for each detector on the later period, recall at your alert budget, and a recommendation of which one to deploy and why.`,
      stretch: [`Train an autoencoder on normal transactions only and use reconstruction error as the score.`, `Test on the last few hours only and see how stable the ranking is.`],
      skills: ['time-based splits', 'extreme imbalance', 'average precision', 'alert budgets', 'isolation forests'] })],

    'time-series': [P({ title: 'Forecast bike rentals', level: 'intermediate', hours: 6, practice: 'bike_rentals_hourly.csv',
      goal: `Forecast hourly bike rentals for the next week, beat the seasonal naive baseline, and give prediction intervals that hold up.`,
      data: `The Bike Sharing dataset (UCI): hourly counts of Capital Bikeshare rentals in Washington, D.C., in 2011 and 2012, with weather, holiday and working-day columns.`,
      steps: [
        `Plot the series at several scales (two years, one month, one week) and name the patterns: daily, weekly and yearly cycles, growth, dips on bad-weather days.`,
        `Hold out the last 8 weeks. Set up rolling-origin evaluation: forecast one week ahead, move forward one week, repeat.`,
        `Baselines: the count at the same hour one week earlier (seasonal naive), and the average of the last four weeks at that hour.`,
        `A feature model: hour, day of the week, holiday, temperature, humidity and the count one week earlier, with gradient-boosted trees. Use only features that would be known when the forecast is made.`,
        `Report MAE and MASE for every model; a MASE below 1 means the model beats the seasonal naive baseline.`,
        `Add 80% prediction intervals with quantile regression (gradient boosting with a quantile loss for the 10th and 90th percentiles), and check that about 80% of actual values fall inside them.`],
      done: `A table of MAE and MASE against the baselines over the rolling folds, a plot of one forecast week with its interval, and the measured coverage of the interval.`,
      stretch: [`Treat the weather as unknown (use seasonal averages) and measure how much accuracy depends on it.`, `Fit a classical model (ETS or SARIMA in statsmodels) to the daily totals.`],
      skills: ['seasonality', 'rolling-origin evaluation', 'features without leakage', 'MASE', 'prediction intervals'] })],

    'stat-inference': [P({ title: 'Plan and analyze an A/B test', level: 'intermediate', hours: 4,
      goal: `Plan, simulate and analyze an online experiment: sample size, the test, confidence intervals, and the checks that catch a broken experiment.`,
      data: `Simulated with NumPy, so you know the true effect: two groups, a baseline conversion rate of 10%, and a true lift you choose, for example half a percentage point.`,
      steps: [
        `Compute the sample size per group needed to detect a lift of 0.5 percentage points with 80% power at a 5% significance level. Check it by simulation: run 1,000 experiments and count how often the test detects the lift.`,
        `Simulate one experiment and analyze it: the difference in conversion rates, a two-proportion z-test, and a 95% confidence interval for the difference.`,
        `Add a sample ratio mismatch check: with a planned 50/50 split, is the observed split consistent with it (a chi-square test)?`,
        `Show the peeking problem: simulate experiments with no true effect, test every day, and stop at the first p-value below 0.05. How often does that produce a false positive?`,
        `Simulate 20 metrics with no true effect and count how many look significant. Apply a Bonferroni or Benjamini-Hochberg correction.`,
        `Write a one-page decision memo: the estimate, its interval, the checks you ran, and whether to ship.`],
      done: `Code that plans and analyzes an experiment, simulations showing the false positive rate with and without peeking, and a decision memo.`,
      stretch: [`Use CUPED (adjusting for a pre-experiment covariate) and measure how much narrower the confidence interval gets.`, `Analyze the same data with a Bayesian beta-binomial model and compare the conclusions.`],
      skills: ['power and sample size', 'confidence intervals', 'sample ratio mismatch', 'peeking', 'multiple testing'] })],

    'pandas-eda': [P({ title: 'An exploratory report on penguins', level: 'starter', hours: 3,
      goal: `Explore a small real dataset with pandas, answer five concrete questions with numbers and plots, and write the findings up as a short report.`,
      data: `The Palmer Penguins data: 344 penguins of 3 species from three islands near Palmer Station, Antarctica, with bill, flipper and body mass measurements and sex. Load it with \`seaborn.load_dataset("penguins")\`.`,
      steps: [
        `Check the shape, the column types and the missing values. Decide what to do with rows that have missing measurements, and write the decision down.`,
        `Summaries: \`describe()\`, counts per species and island, and a \`groupby\` table of mean measurements by species and sex.`,
        `Distributions: a histogram of body mass for each species, and bill length against bill depth coloured by species.`,
        `Answer five questions, for example: Which species is heaviest? Are males heavier in every species, and by how much? Which island has which species? Which two measurements separate the species best?`,
        `Look for Simpson's paradox: compute the correlation between bill length and bill depth over all penguins, then within each species.`,
        `Write a one-page report: each question, its answer with a number, and the plot that shows it.`],
      done: `A notebook or script that runs from top to bottom without errors, and a short report that answers five questions.`,
      stretch: [`Fit a simple classifier on the two best measurements and plot its decision regions.`, `Turn the notebook into a script with functions that saves every figure.`],
      skills: ['loading and checking data', 'groupby and aggregation', 'plots that answer questions', 'writing up findings'] })],

    'sql-data-engineering': [P({ title: 'A small analytics pipeline in SQL', level: 'starter', hours: 4,
      goal: `Answer business questions about a music store with SQL, then build a repeatable pipeline that loads new data and refreshes a summary table.`,
      data: `The Chinook sample database (a digital music store with customers, invoices, tracks, albums and artists), available as a SQLite file from the chinook-database project on GitHub.`,
      steps: [
        `Open the database with Python's built-in \`sqlite3\` module and list the tables and their columns.`,
        `Write queries for: revenue by country; the 10 artists with the most revenue; revenue per month; customers with no purchase in the last year of the data.`,
        `Use window functions to rank each customer's invoices and to compute a running total of revenue by month.`,
        `Create a \`monthly_revenue\` summary table with a statement that is safe to run again (delete and re-insert the month, or \`INSERT OR REPLACE\`).`,
        `Write a Python script that loads a CSV of new invoices into a staging table, checks it (known customer, positive total), moves the valid rows into the main tables and refreshes the summary.`,
        `Add indexes on the columns your queries filter and join on, and compare the plans with \`EXPLAIN QUERY PLAN\`.`],
      done: `A script you can run twice on the same input without creating duplicates, and answers to the four questions.`,
      stretch: [`Load the summary into pandas and plot revenue per month.`, `Port the pipeline to DuckDB and compare.`],
      skills: ['joins and aggregation', 'window functions', 'repeatable loads', 'data validation', 'indexes'] })],

    'mlp': [P({ title: 'Handwritten digits with a network you wrote', level: 'intermediate', hours: 8,
      goal: `Train a neural network with one hidden layer on MNIST using only NumPy, with your own forward pass, backward pass and optimizer, and reach about 97% test accuracy.`,
      data: `MNIST: 70,000 grayscale images of handwritten digits, 28 by 28 pixels, 60,000 for training and 10,000 for testing. Load it with \`sklearn.datasets.fetch_openml("mnist_784", version=1)\` or from torchvision.`,
      steps: [
        `Scale the pixels to the range 0 to 1, flatten each image to 784 numbers, and keep 5,000 training images aside for validation.`,
        `Implement a network with 128 ReLU hidden units and a softmax output: the forward pass, the cross-entropy loss and the backward pass, by hand.`,
        `Check your gradients against finite differences on a tiny batch before training. This catches most bugs.`,
        `Train with mini-batch SGD (batch size 64, learning rate about 0.1) for 10 epochs, and plot training and validation loss and accuracy per epoch.`,
        `Add momentum or Adam, and L2 regularization, and compare the curves.`,
        `Show 20 misclassified test digits with their predicted and true labels.`],
      done: `Learning curves for at least two optimizers, a test accuracy of about 97% or more, and a gradient check that passes.`,
      stretch: [`Rewrite the network in PyTorch and confirm you get the same loss on the same batch.`, `Add a second hidden layer and dropout.`],
      skills: ['backpropagation by hand', 'gradient checking', 'mini-batch training', 'optimizers', 'learning curves'] })],

    'cnn': [P({ title: 'An image classifier for CIFAR-10', level: 'intermediate', hours: 8,
      goal: `Train a convolutional network in PyTorch that sorts small colour images into 10 classes, and raise its accuracy with augmentation and a better training schedule.`,
      data: `CIFAR-10: 60,000 colour images of 32 by 32 pixels in 10 classes (airplane, car, bird, cat and others), 50,000 for training and 10,000 for testing. \`torchvision.datasets.CIFAR10\` downloads it.`,
      steps: [
        `Normalize each colour channel with the mean and standard deviation of the training set, and hold out 5,000 training images for validation.`,
        `Start small: three blocks of convolution, batch normalization, ReLU and max pooling, then a linear layer. Train for 20 epochs with SGD and momentum, or AdamW.`,
        `Plot training and validation accuracy. If training accuracy is far above validation accuracy, the model is overfitting.`,
        `Add augmentation (random crops with padding, horizontal flips) and a learning-rate schedule (one-cycle or cosine), and compare.`,
        `Make a table of accuracy per class and a confusion matrix. Which classes get mixed up?`,
        `Fine-tune a pretrained ResNet-18 from torchvision on the same data, and compare accuracy and training time.`],
      done: `A small network at around 85% test accuracy or better, a comparison with the fine-tuned ResNet, and a confusion matrix with a sentence about the main confusions.`,
      stretch: [`Visualize the first-layer filters and the feature maps for one image.`, `Use Grad-CAM to see which pixels drive a prediction.`],
      skills: ['convolutional networks', 'data augmentation', 'learning-rate schedules', 'transfer learning', 'training on a GPU'] })],

    'tokenization': [P({ title: 'Train your own BPE tokenizer', level: 'intermediate', hours: 4,
      goal: `Implement byte-pair encoding from scratch, train it on a book, and compare it with a production tokenizer.`,
      data: `Any long public-domain text from Project Gutenberg (a novel of 500,000 characters or more), saved as plain UTF-8 text.`,
      steps: [
        `Start from the 256 byte values, so that any string can be encoded. Count adjacent pairs, merge the most frequent pair into a new token, and repeat until you have 1,000 tokens.`,
        `Write \`encode\` (apply the merges in the order they were learned) and \`decode\` (join the bytes), and test that decoding an encoding gives back the original, on random strings with accents and non-Latin scripts.`,
        `Measure compression on held-out chapters: characters per token for vocabularies of 300, 1,000 and 3,000 tokens.`,
        `Split text into words and punctuation with a regular expression before counting pairs, as GPT-2 does, and see how the learned tokens change.`,
        `Compare with a trained tokenizer (\`tiktoken\` or Hugging Face \`tokenizers\`): tokens per word on English prose, on code and on another language.`],
      done: `A tokenizer that round-trips any string, a table of compression by vocabulary size, and the comparison with a production tokenizer.`,
      stretch: [`Speed up training by updating pair counts after each merge instead of recounting everything.`, `Add special tokens, such as an end-of-text marker, that are never split.`],
      skills: ['byte-level BPE', 'encoding and decoding', 'compression ratio', 'pre-tokenization'] })],

    'text-embeddings': [P({ title: 'Semantic search over your own notes', level: 'intermediate', hours: 5,
      goal: `Build a search tool that finds the passages in your own notes or documents that answer a question, even when they use different words, and measure how well it works.`,
      data: `Your own Markdown or text files (notes, a documentation folder or a set of articles), at least 50 documents. Use the \`sentence-transformers\` library with a small model such as \`all-MiniLM-L6-v2\`, which runs on a laptop CPU.`,
      steps: [
        `Split the documents into chunks of 200 to 400 words with a small overlap, keeping the file name and heading with each chunk.`,
        `Embed every chunk once and save the vectors (a NumPy file is enough at this size) together with the chunk text.`,
        `Search: embed the query, compute cosine similarity with every chunk, and return the top 5 with their sources.`,
        `Write 20 test questions, each with the chunk that answers it. Measure recall at 5 (how often the right chunk is in the top 5) and mean reciprocal rank.`,
        `Compare with keyword search (BM25, from the \`rank-bm25\` package) and with a hybrid that combines both scores.`,
        `Try two chunk sizes and keep the one with better recall.`],
      done: `A command-line tool that answers a query in under a second, and recall at 5 and MRR for embedding, keyword and hybrid search on your 20 questions.`,
      stretch: [`Add a cross-encoder that re-ranks the top 20 results.`, `Store the vectors in FAISS or Chroma and compare speed.`],
      skills: ['chunking', 'embeddings', 'cosine similarity', 'retrieval metrics', 'hybrid search'] })],

    'mini-gpt': [P({ title: 'Train a small GPT on text you choose', level: 'advanced', hours: 10,
      goal: `Train the lesson's GPT on a corpus you pick, tune it until its samples clearly follow the style of the corpus, and measure progress with validation loss.`,
      data: `A plain-text corpus of 1 to 10 MB: the Tiny Shakespeare file (about 1.1 million characters) used in many tutorials, your own writing, or the source code of a project.`,
      steps: [
        `Start at the character level: build the vocabulary, encode the text, and split it 90/10 into training and validation.`,
        `Port the lesson's model to PyTorch if you have not already, and train a small configuration (4 layers, 4 heads, 128-dimensional embeddings, a context of 128 characters).`,
        `Log training and validation loss every few hundred steps, and stop when validation loss stops improving.`,
        `Generate samples at temperatures 0.5, 0.8 and 1.0, and with top-k sampling. Describe how they differ.`,
        `Change one thing at a time (layers, embedding size, context length) and record validation loss against the number of parameters.`,
        `Switch to a BPE tokenizer (your own, or GPT-2's) and compare loss per character rather than per token, so the numbers are comparable.`],
      done: `A loss curve, a table of validation loss for three or more model sizes, and samples at three temperatures.`,
      stretch: [`Add learning-rate warm-up with cosine decay and gradient clipping, and compare the stability of training.`, `Fine-tune the trained model on a smaller second corpus and watch its style shift.`],
      skills: ['training a transformer', 'validation loss', 'sampling', 'scaling experiments', 'choosing a tokenizer'] })],

    'llm-evaluation': [P({ title: 'An evaluation suite for an LLM feature', level: 'intermediate', hours: 6,
      goal: `Write a test set and an automatic grader for one LLM feature (for example, summarizing support tickets), so that every change to the prompt or the model is measured instead of judged by eye.`,
      data: `50 to 100 real or realistic inputs for the feature, each with what a good answer must contain. Any language model you can call: a hosted API, or an open model run locally with a tool such as Ollama or Hugging Face \`transformers\`.`,
      steps: [
        `Write down what success means: 3 to 5 criteria a good output must meet (correct facts, required fields present, a length limit, tone).`,
        `Collect inputs that cover normal cases, hard cases and cases that must be refused or escalated. Keep 20% as a held-out set you rarely look at.`,
        `Grade with code wherever you can: schema checks, required keywords, length, exact answers.`,
        `For criteria that code cannot check, write a judge prompt with a rubric and examples. Grade 30 outputs yourself and measure how often the judge agrees with you.`,
        `Run the suite on two prompt versions and two models, and report pass rates per criterion with confidence intervals; with 50 cases, a difference of a few points is noise.`,
        `Make the whole suite run with one command, and save results with the prompt version, model and date.`],
      done: `One command that scores a prompt and model on every criterion, a measured judge agreement rate, and a comparison of two versions.`,
      stretch: [`Add regression cases: inputs that once failed and must keep passing.`, `Record cost and latency per case next to quality.`],
      skills: ['success criteria', 'test set design', 'grading with code', 'validating an LLM judge', 'comparing versions'] })],

    'agents': [P({ title: 'A data analyst agent for CSV files', level: 'advanced', hours: 8,
      goal: `Build an agent that answers questions about a CSV file by writing and running pandas code, checking its results, and stopping when it has an answer.`,
      data: `A CSV file you know well (for example the penguins data, or invoices exported from the Chinook database), and a language model that supports tool calling, hosted or local.`,
      steps: [
        `Define two tools with clear descriptions and JSON schemas: \`describe_data\` (columns, types and a few rows) and \`run_pandas\` (runs code against a DataFrame named df and returns the printed output or the error).`,
        `Write the loop: send the question and the tool definitions, run any tool calls, send back the results, and repeat until the model answers in text or a step limit (for example 8) is reached.`,
        `Run generated code in isolation: a subprocess with a timeout, no network access and a copy of the data. Never run it with \`exec\` inside your own process.`,
        `Return errors to the model as tool results, so it can fix its own code.`,
        `Write 20 questions whose answers you have computed yourself, and measure how many the agent gets right, how many steps it takes and what it costs.`,
        `Log every step (model message, tool call, result) to a file you can read when something goes wrong.`],
      done: `An agent that answers at least 15 of your 20 questions correctly within the step limit, with a log of every run.`,
      stretch: [`Add a plotting tool that saves a chart and returns its file path.`, `Ask the agent to state its assumptions before answering, and test whether that reduces wrong answers.`],
      skills: ['tool definitions', 'the agent loop', 'sandboxing generated code', 'recovering from errors', 'evaluating agents'] })],

    'recommender-systems': [P({ title: 'Movie recommendations from ratings', level: 'intermediate', hours: 6,
      goal: `Recommend movies to users from their past ratings, compare a popularity baseline with collaborative filtering, and evaluate with ranking metrics.`,
      data: `MovieLens 100K from GroupLens: 100,000 ratings (1 to 5 stars) from 943 users on 1,682 movies, with titles and genres.`,
      steps: [
        `Split by time for each user: the last 20% of each user's ratings form the test set, so you predict the future from the past.`,
        `Baseline: recommend the most popular movies the user has not rated. Measure precision and recall at 10, counting test movies rated 4 or 5 as relevant.`,
        `Item-based collaborative filtering: cosine similarity between the rating vectors of movies; score each unseen movie by its similarity to movies the user liked.`,
        `Matrix factorization: learn user and movie vectors (20 to 50 dimensions) with biases, by SGD or alternating least squares with L2 regularization, tuned on a validation split.`,
        `Compare all three on precision and recall at 10, and on coverage (how many different movies get recommended to anyone).`,
        `Show the recommendations for three users next to the movies they rated highly, and judge whether they make sense.`],
      done: `A table of precision at 10, recall at 10 and coverage for three methods, and sample recommendations for three users.`,
      stretch: [`Handle a new user: ask for five ratings and recommend from those.`, `Use genres to recommend new movies that have no ratings yet.`],
      skills: ['explicit feedback', 'time-based splits', 'matrix factorization', 'ranking metrics', 'cold start'] })],

    'reinforcement-learning': [P({ title: 'Teach an agent to cross a frozen lake', level: 'intermediate', hours: 5,
      goal: `Solve a small environment with tabular Q-learning, then see why a table stops working when the state is continuous.`,
      data: `The Gymnasium library (\`pip install gymnasium\`): FrozenLake-v1, a 4 by 4 grid of slippery ice with holes, and CartPole-v1, balancing a pole on a moving cart.`,
      steps: [
        `Run 1,000 episodes of a random policy on FrozenLake and record the success rate as the baseline.`,
        `Implement Q-learning with an epsilon-greedy policy: a 16 by 4 table, a learning rate, a discount factor of 0.99, and epsilon decaying from 1 to 0.05.`,
        `Plot the success rate during training as a moving average over 100 episodes. Solve the non-slippery version first, then the slippery one.`,
        `Print the learned greedy policy as arrows on the grid, and explain the moves that look strange (on slippery ice the agent aims away from holes).`,
        `Move to CartPole: split each of the four continuous state variables into bins and run the same Q-learning. How many bins work, and what goes wrong with too many?`],
      done: `A FrozenLake agent with a success rate far above the random baseline, a plot of its learning, and a short note on why CartPole needs function approximation.`,
      stretch: [`Replace the table with a small neural network (DQN, with a replay buffer and a target network) for CartPole.`, `Compare SARSA with Q-learning on the slippery lake.`],
      skills: ['Markov decision processes', 'Q-learning', 'exploration', 'reward curves', 'discretizing states'] })],

    'multimodal-models': [P({ title: 'Sort photos by description, with no training', level: 'intermediate', hours: 4,
      goal: `Use a pretrained CLIP model to sort photos into categories you describe in words, and find out where zero-shot classification works and where it fails.`,
      data: `100 to 300 of your own photos (or a public set such as a few classes of Imagenette), and the \`openai/clip-vit-base-patch32\` model through Hugging Face \`transformers\`.`,
      steps: [
        `Label 100 photos yourself into 5 to 8 categories (for example food, people, pets, landscapes, documents). These labels are only for evaluation.`,
        `Embed every image once with the CLIP image encoder, and every category with the text encoder, using a prompt such as "a photo of food".`,
        `Classify each photo by the highest cosine similarity, and measure accuracy and the confusion matrix against your labels.`,
        `Try several prompt templates per category and average their text embeddings. Does accuracy improve?`,
        `Search the other way round: type a description ("a dog on a beach at sunset") and return the closest photos.`,
        `Collect the failures and describe them: similar categories, unusual angles, text inside images.`],
      done: `Zero-shot accuracy with one prompt and with averaged prompts, a working text-to-photo search, and a list of failure types.`,
      stretch: [`Train a logistic regression on the CLIP image embeddings with 10 labelled photos per class (a linear probe) and compare with zero-shot.`, `Cluster the image embeddings and see which groups appear without any labels.`],
      skills: ['contrastive embeddings', 'zero-shot classification', 'prompt templates', 'linear probes', 'search across modalities'] })],

    'gnn': [P({ title: 'Classify research papers by their citations', level: 'intermediate', hours: 5,
      goal: `Predict the topic of a paper from its words and from the papers it cites, and measure how much the graph adds to the words alone.`,
      data: `The Cora citation network: 2,708 machine learning papers in 7 topics, 5,429 citation links, and a 1,433-word bag-of-words vector for each paper. PyTorch Geometric loads it with \`Planetoid(root, "Cora")\`, including the standard split with 140 labelled training papers.`,
      steps: [
        `Baseline without the graph: logistic regression or an MLP on the word vectors, trained on the 140 labelled papers. Record the test accuracy.`,
        `Implement one graph convolution layer yourself: add self-loops, normalize the adjacency matrix, and multiply it with the features and a weight matrix.`,
        `Train a two-layer GCN (your layer or \`GCNConv\`) with dropout and weight decay, and report test accuracy over 10 random seeds as a mean and standard deviation.`,
        `Try label propagation alone, with no features: spread the 140 known labels along the citation links. How close does it get?`,
        `Plot the learned node embeddings in two dimensions (t-SNE or PCA), coloured by topic.`],
      done: `A table of test accuracy for the MLP, label propagation and the GCN, each over 10 seeds, and an embedding plot.`,
      stretch: [`Replace the GCN with a graph attention network (\`GATConv\`) and compare.`, `Remove a random half of the edges and measure how accuracy drops.`],
      skills: ['graph data', 'message passing', 'semi-supervised learning', 'variance across seeds', 'visualizing embeddings'] })],

    'mlops': [P({ title: 'Ship a model as a monitored service', level: 'advanced', hours: 10,
      goal: `Take a trained model (for example the income or churn model from an earlier project) to a small production service with an API, tests, a container, and monitoring that would notice drift.`,
      data: `A model you trained earlier, saved with \`joblib\`, plus its training data as the reference for drift checks.`,
      steps: [
        `Write a FastAPI app with a \`/predict\` endpoint whose input is validated by a Pydantic model (types and allowed ranges), and a \`/health\` endpoint.`,
        `Log every request and prediction, without personal data, with the model version and a timestamp.`,
        `Write tests: the model gives a known output for a known input; invalid input is rejected; the API works end to end (FastAPI's test client).`,
        `Package the service in a Docker image with pinned package versions, run it, and send it requests.`,
        `Add a GitHub Actions workflow that runs the tests on every push.`,
        `Write a drift report: compare the distribution of each input feature in the logs with the training data (population stability index or a Kolmogorov-Smirnov test) and flag the features that moved.`],
      done: `A container that serves predictions, tests that run in CI, and a drift report that flags a feature when you deliberately send shifted data.`,
      stretch: [`Add a shadow model: serve version 1, log what version 2 would have predicted, and compare.`, `Track experiments and model versions with MLflow.`],
      skills: ['model serving', 'input validation', 'testing ML code', 'containers and CI', 'drift monitoring'] })],

    'quantization': [P({ title: 'Quantize a small language model and measure the cost', level: 'advanced', hours: 6,
      goal: `Run an open language model at 16, 8 and 4 bits, and measure what each step saves in memory and time and what it costs in quality.`,
      data: `A small open model (0.5 to 3 billion parameters) from Hugging Face, and a held-out text for perplexity, such as the WikiText-2 test set. Tools: \`transformers\` with \`bitsandbytes\` on an NVIDIA GPU, or \`llama.cpp\` with GGUF files on a CPU or a Mac.`,
      steps: [
        `Load the model in 16-bit and record the memory used, the tokens per second when generating 200 tokens, and the perplexity on the test text with a fixed context length.`,
        `Repeat with 8-bit and 4-bit weights (in llama.cpp, for example the Q8_0 and Q4_K_M formats).`,
        `Put the results in one table: size, speed and perplexity. Compute the size you expect from the bits per weight and compare it with what you measured.`,
        `Run 10 fixed prompts (facts, arithmetic, code) at each precision with greedy decoding, and compare the answers side by side.`,
        `Write a recommendation: which precision would you run on a laptop with 8 GB of memory, and which on a server GPU, and why?`],
      done: `A table of size, speed and perplexity at three precisions, the side-by-side answers, and a recommendation.`,
      stretch: [`Try a 3-bit format and find where quality drops sharply.`, `Measure accuracy on a few hundred multiple-choice questions as well as perplexity.`],
      skills: ['weight quantization', 'perplexity', 'measuring memory and speed', 'quality trade-offs'] })],

    'speech-audio': [P({ title: 'Recognize spoken command words', level: 'intermediate', hours: 6,
      goal: `Train a small model that recognizes one-second spoken words such as "yes", "no", "up" and "down", the task behind wake words on devices.`,
      data: `Google's Speech Commands dataset (version 2): about 105,000 one-second recordings of 35 words by thousands of speakers. \`torchaudio.datasets.SPEECHCOMMANDS\` downloads it, with the official validation and test lists.`,
      steps: [
        `Pick 10 target words, plus an "unknown" class (all the other words) and a "silence" class (clips cut from the background noise files).`,
        `Turn each clip into a log-mel spectrogram (for example 40 mel bands, 25 ms windows and 10 ms hops) and look at a few as images.`,
        `Train a small CNN on the spectrograms with the official splits, which keep each speaker in only one split.`,
        `Add augmentation: random time shifts, background noise mixed in at random volumes, and masking of random time and frequency bands (SpecAugment).`,
        `Report accuracy and a confusion matrix, and listen to some of the mistakes.`,
        `Measure the model's size and how long it takes to classify one clip on your CPU.`],
      done: `Test accuracy on the official split, a confusion matrix, and the model's size and latency.`,
      stretch: [`Record yourself saying the words and test the model on your voice.`, `Run it on a live microphone stream with a sliding window.`],
      skills: ['audio features', 'spectrograms', 'splits by speaker', 'audio augmentation', 'latency'] })],

    'detection-segmentation': [P({ title: 'Count objects in your own images', level: 'intermediate', hours: 6,
      goal: `Use a pretrained object detector to count objects (cars on a street, people in a room, birds at a feeder) in your own images, and measure how accurate the counts are.`,
      data: `50 to 100 images you take or collect, and a detector pretrained on COCO: \`torchvision.models.detection.fasterrcnn_resnet50_fpn\`, or a YOLO model from the \`ultralytics\` package.`,
      steps: [
        `Count the objects in 30 images by hand. This is your ground truth.`,
        `Run the detector, keep the boxes of your class above a confidence threshold, and draw them on the images.`,
        `Plot the count error (predicted minus true) against the threshold, and choose the threshold with the smallest mean absolute error.`,
        `Draw boxes by hand on 20 images (a labelling tool such as Label Studio or CVAT helps) and compute precision and recall at an IoU of 0.5.`,
        `Sort the failures: small or distant objects, overlapping objects, unusual views. Which kind is most common?`],
      done: `The mean absolute count error on your 30 images at the chosen threshold, precision and recall at an IoU of 0.5, and a ranked list of failure types.`,
      stretch: [`Fine-tune the detector on 100 labelled images of your scene and measure the improvement.`, `Use a segmentation model to measure how much of the image each object covers.`],
      skills: ['pretrained detectors', 'confidence thresholds', 'IoU', 'precision and recall for detection', 'error analysis'] })]
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
