## Simpplrai

To support custom code specific Simpplr Analytics functionality

## Chronology of updates

* Support for Page Reco reranking based on People/Site followed
* Support for common haystack object for all indexes
* 30-Nov-2021 : Merged changes related to git deployment for Stage environment
* 01-Dec-2021 : Removed Redis dependency in eapl_ws and log level increased from debug to info for requests
* 11-Jan-2022 : Code Version - 1.0.1 : Added support code version variable for testing and versioning
* 21-Jan-2022 : Code Version - 1.0.2 Improvement html to text conversion, improvments in text preprocessing and tagging
  score refine Update the Semantic drop duplicate model to sentence-transformers/all-MiniLM-L6-v2
* 24-Jan-2022 : Code Version - 1.0.3 Modified event filtering only based on expiry date, Extended filters to support
  deleted content filtering, improved error handing and logging if 0 recommendation are generated after filtering or no
  content is recommended from cold start
* 25-Jan-2022 : Code Version - 1.0.4 Added checks to handle empty sparse matrix if no user have more than 5 content as
  recommendations
* 14-Feb-2022 : Code Version - 1.0.5 Content recommendation code refactoring.
  Simplified the flow and heavy lifting is completely done in content reco setup API itself
  Performance optimization
  Added support for multiple reco method and content deduplication
  Support for restricting the content to only 2 categories max
  For new user added support with default cold start user recommendations
  Elimination of viewed content for user having <=5 content interactions
  Optimization of content reco indexing and code updated to handle bulk API write
  Variable renaming convention
  Optimization of ranking for all the features
* 21-Feb-2022 : Code Version - 1.0.6 
  Doc strings for Topic suggestion and Content Recommendations
* 22-Feb-2022 : Code Version - 1.0.7
    Support of Home Carousel based reco
    Support for must read content reco
* 03-Mar-2022 : Code Version - 1.0.8
  Changed logging of redis to info level for seeing the data in logs.
* 04-Mar-2022 : Code Version - 1.0.9
  Fix for 1 user 1 document test case
  Fix on output structure for email
  eapl_nlp #451: Handle exception with top_n=50
* 07-Mar-2022 : Code Version - 1.0.10
  Handling event expiry for must read and case sensitivity with content_type
* 11-Mar-2022 : Code Version - 1.0.11
  Fix for old must read content removed variable handling
* 17-Mar-2022 : Code Version - 1.0.12
  Updated Dev Jenkin file with new credentials and configuration
* 21-Mar-2022 : Code Version - 1.0.13
  No code changes. New version to test the deployment pipeline on new dev instance.
  Extended html cleanup on title field for topic suggestion.
* 21-Mar-2022 : Code Version - 1.0.14
  Code segration to relevant APIs. Shall simplify the imports and reduce overheads for background jobs
* 28-Mar-2022 : Code Version - 1.0.15
  Removed dependency on CollaborationGroupMember table.
* 31-Mar-2022 : Code Version - 1.0.16
  Test case enhancements for regression check
  Upgrade in implicit library and optimization to eliminate redundant sparse matrix
* 04-Apr-2022 : Code Version - 1.0.17
  Limited the Simpplr__Content__Interaction__c table data load to last one year.
* 26-Apr-2022 : Code Version - 1.0.18
  Added createdAt and updatedAt timestamp feature to Simpplr__Content_Recommendation_Count__c table
* 03-May-2022 : Code Version - 1.0.19
  Controlled the recommendations generated from Carousel.
  