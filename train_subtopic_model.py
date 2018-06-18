"""
python train_subtopic_model.py <topics_id> <focal_set_name> <tag_sets_id>
(e.g. python train_subtopic_model.py 1375 press-impact 15765103)

Trains a Multinomial Naive Bayes classifier on set of stories in csv file.
Optional: Run trained model on all stories within the topic and...
 - Tag each story as part of the subtopic or not (must specifiy tag_sets_id as second arg)
 - Generate csv file with results for each story

Must set MC_API_KEY environment variable
"""


import logging
import csv
import time
import sys
import os
import mediacloud
import codecs
import re
import numpy as np

from mediacloud.tags import StoryTag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.externals import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TAG_STORIES = False
GEN_CSV = False

try:
    MC_API_KEY = os.environ['MC_API_KEY']
except KeyError:
    print 'You need to define the MC_API_KEY environment variable.'
    sys.exit(0)

mc = mediacloud.api.AdminMediaCloud(MC_API_KEY)

TOPIC_ID = sys.argv[1]

FOCAL_SET_NAME = sys.argv[2]

if TAG_STORIES:
    TAG_SETS_ID = sys.argv[3]

ID_FILE = 'train/ids/{}-ids.csv'.format(FOCAL_SET_NAME)

MODEL_FILENAME_TEMPLATE = 'topic-{}-{}.pkl' # topic id, model_name
VECTORIZER_FILENAME_TEMPLATE = 'topic-{}-{}-vec.pkl' # topic id, model_name

MIN_DF = 0.1
MAX_DF = 0.9


def download_stories_from_csv(filepath):
    """
    @param: filepath, string
    Returns list of story ids and corresponding labels as training data
    """
    logger.info('Downloading sample stories from topic...')
    start = time.time()

    acceptable_column_names = ['stories_id', 'label']
    with open(filepath, 'rU') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = acceptable_column_names
        stories = []
        labels = []
        reader.next()   # skip column headers
        for row_num, row in enumerate(reader):
            stories_id = row['stories_id']
            label = row['label']

            # validate row entries
            try:
                stories_id = int(stories_id)
            except Exception as e:
                err_msg = "Couldn't process row number {}: invalid stories_id".format(str(row_num + 2))
                logger.error(err_msg)
                raise Exception(err_msg)
            try:
                label = int(label)
            except Exception as e:
                err_msg = "Couldn't process row number {}: label must be 0 or 1".format(str(row_num + 2))
                logger.error(err_msg)
                raise Exception(err_msg)
            if label != 1 and label != 0:
                err_msg = "Couldn't process row number {}: label must be 0 or 1".format(row_num + 2)
                logger.error(err_msg)
                raise Exception(err_msg)

            stories.append(mc.story(stories_id, sentences=True))
            labels.append(label)

    end = time.time()
    logger.info('Download time: {}'.format(str(end-start)))

    return stories, labels


def process_stories_text(stories_list):
    topic_stories_text = []
    for i, story in enumerate(stories_list):
        topic_stories_text.append('')
        for sentence in story['story_sentences']:
            sent = re.sub(r'[^\w\s-]', '', sentence['sentence'])
            sent = re.sub(r'[\s-]', ' ', sent)
            topic_stories_text[i] += (sent.lower() + ' ')

    return topic_stories_text


def tag_topic_stories(topic_stories, pred_labels, pred_probs, batch_size=20):
    # Create tags
    pos_tag_name = 'matching-{}'.format(FOCAL_SET_NAME)
    pos_tag_label = FOCAL_SET_NAME
    pos_tag_description = 'Story was classified as part of the subtopic \'{}\''.format(pos_tag_label)
    neg_tag_name = 'not-matching-{}'.format(FOCAL_SET_NAME)
    neg_tag_label = 'Not {}'.format(FOCAL_SET_NAME)
    neg_tag_description = 'Story was classified as part of the subtopic \'{}\''.format(neg_tag_label)
    pos_tag = mc.createTag(TAG_SETS_ID, pos_tag_name, pos_tag_label, pos_tag_description)
    neg_tag = mc.createTag(TAG_SETS_ID, neg_tag_name, neg_tag_label, neg_tag_description)

    # Tag topic stories in batches
    max_len = len(predicted_labels)
    for i in range(0, max_len, batch_size):
        batch_labels = predicted_labels[i:min(i+batch_size, max_len)]
        story_tags = []
        for j, label in enumerate(batch_labels):
            story_id = topic_stories[i+j]['stories_id']
            if (label == 1.0):
                story_tags.append(StoryTag(story_id, tags_id=pos_tag['tags_id']))
            else:
                story_tags.append(StoryTag(story_id, tags_id=neg_tag['tags_id']))
        mc.tagStories(tags=story_tags)  # uncomment to tag


def generate_output(topic_stories, pred_labels, pred_probs):
    with codecs.open('topic-{}-{}-results.csv'.format(TOPIC_ID, FOCAL_SET_NAME), 'w', 'utf-8') as csvfile:
        # write header
        fieldnames = ['stories_id', 'title', 'media_source', 'url', 'label', 'probability\n']
        csvfile.write(','.join(fieldnames))

        for i, story in enumerate(topic_stories):
            story_id = str(story['stories_id'])
            title = re.sub(r'[,\s]', ' ', story['title'])
            media_source = str(story['media_id']).strip()
            url = re.sub(r'[,\s]', '', story['url'])
            label = str(pred_labels[i]).strip()
            prob = str(pred_probs[i]).strip()
            csvfile.write(','.join([story_id, title, media_source, url, label, prob+'\n']))

if __name__ == "__main__":

    # Load stories_ids and labels from csv
    stories, labels = download_stories_from_csv(ID_FILE)
    logger.info('Training model on {} stories'.format(len(stories)))

    # Load and vectorize text data
    stories_text = process_stories_text(stories)
    logger.info('Vectorizing data...')
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=MIN_DF, max_df=MAX_DF)
    vectorizer.fit(stories_text)
    X_train = vectorizer.transform(stories_text)
    y_train = np.asarray(labels)
    logger.info('Number of Training Examples: {}'.format(X_train.shape))
    logger.info('Number of Labels: {}'.format(y_train.shape))

    # Train model
    logger.info('Training model...')
    clf = MultinomialNB()
    model = clf.fit(X_train, y_train)
    logger.info('Training Accuracy: {}'.format(model.score(X_train, y_train)))

    # K-fold Cross Validation (stratified)
    logger.info('Cross validating...')
    skf = StratifiedKFold(n_splits=5)
    test_prec_scores = []
    test_rec_scores = []
    for train_index, test_index in skf.split(X_train, y_train):
        # train
        X_train_val, X_test_val = X_train[train_index], X_train[test_index]
        y_train_val, y_test_val = y_train[train_index], y_train[test_index]
        clf = MultinomialNB()
        model = clf.fit(X_train_val, y_train_val)
        # test
        test_prec_score = precision_score(y_test_val, model.predict(X_test_val))
        test_rec_score = recall_score(y_test_val, model.predict(X_test_val))
        # update scores
        test_prec_scores.append(test_prec_score)
        test_rec_scores.append(test_rec_score)

    # Mean precision and recall
    logger.info('Average Test Precision: {}'.format(np.mean(test_prec_scores)))
    logger.info('Std Dev: {}'.format(np.std(test_prec_scores)))
    logger.info('Average Test Recall: {}'.format(np.mean(test_rec_scores)))
    logger.info('Std Dev: {}'.format(np.std(test_rec_scores)))

    # Test on topic:
    if TAG_STORIES or GEN_CSV:

        logger.info('Downloading stories from topic...')
        last_processed_id = 0
        page = 0
        more_stories = True
        topic_stories = []
        start = time.time()
        # while len(topic_stories) < 300:
        while more_stories:
            logger.info('Page {}'.format(page))
            topic_stories_page = mc.storyList(solr_query='{~ topic:'+TOPIC_ID+'}', sentences=True, last_processed_stories_id=last_processed_id, rows=50)
            more_stories = len(topic_stories_page) > 0
            if len(stories) > 0:
				last_processed_id = topic_stories_page[-1]['processed_stories_id']
            topic_stories += topic_stories_page
            page += 1
        end = time.time()
        logger.info('Downloaded {} stories from topic {}'.format(len(topic_stories), TOPIC_ID))
        logger.info('Time: {}'.format(str(end-start)))

        # Make predictions with model
        topic_stories_text = process_stories_text(topic_stories)
        X_topic = vectorizer.transform(topic_stories_text)
        predicted_labels = model.predict(X_topic).tolist()
        predicted_probs = model.predict_proba(X_topic).tolist()

        if TAG_STORIES:
            tag_topic_stories(topic_stories, predicted_labels, predicted_probs)

        if GEN_CSV:
            generate_output(topic_stories, predicted_labels, predicted_probs)

    # Pickle model
    joblib.dump(model, MODEL_FILENAME_TEMPLATE.format(TOPIC_ID, FOCAL_SET_NAME))
    joblib.dump(vectorizer, VECTORIZER_FILENAME_TEMPLATE.format(TOPIC_ID, FOCAL_SET_NAME))
