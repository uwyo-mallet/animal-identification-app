
packages = c("devtools","shiny", "shinythemes","shinyFiles")

## Now load or install & load all
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
    }
    library(x, character.only = TRUE)
  }
)

if (!require('MLWIC2')) devtools::install_github("haniyeka/MLWIC2") 
library(MLWIC2)

setwd(".")
# Define UI
shinyUI(
  fluidPage(theme = shinytheme("flatly"),
            navbarPage("Animal Identification",
                       tabPanel("Predict",value = "Predict",
                                sidebarPanel(
                                  tags$h3("Predicting Inputs:"),
                                  fileInput("prediction_data_info","Image label CSV file",accept = c(".csv")),
                                  shinyDirButton('prediction_path_prefix', "Image directory", title="Select the parent directory where images are stored"),
                                  selectInput("prediction_Type", label="Choose Prediction:",
                                              choices= list("empty_animal"="empty_animal","species_model"="species_model"), 
                                              selected = "empty_animal"),
                                  selectInput("prediction_modelChoice", label="Choose Moel:",
                                              choices= list("Auto Selection"="AutoSelection","VGG Model"="VGGmodel", "RESNET Model"="Resnetmodel"), 
                                              selected = "Auto Selection"),
                                  shinyDirButton('prediction_model_dir', 'Models directory', title="Find and select the parent folder"),
                                  actionButton("submitbutton","Submit",class="btn btn-primary")
                                  
                                ), # sidebarPanel
                                mainPanel(
                                  h1("Prediction Results"),
                                  h4("Statistical Results"),
                                  verbatimTextOutput("statisticalResults"),
                                  textOutput("predict_command_print")
                                  
                                ) # mainPanel
                                
                       ),
                       tabPanel("Retrain",value = "Retrain", "This panel is intentionally left blank"),
                       navbarMenu("Setting",
                                  tabPanel("First-time Setup",value = "Setup",
                                           sidebarPanel(
                                             shinyDirButton('python_loc', "Python location", title="Select the location of Python. It should be under Anaconda. Just select the folder where it resides in the top half of the menu and press `Select`"),
                                             selectInput("r_reticulate", "Have you already installed packages in aconda environment called `r-reticulate` that you want to keep?
                                                         If you don't know, click `No`",
                                                         choices = c(
                                                           "No" = FALSE,
                                                           "Yes" = TRUE
                                                           )
                                             ),
                                              selectInput("gpu", "Do you have a GPU on your machine that you are planning to use?
                                                           If you don't know, click `No`",
                                                           choices = c(
                                                          "No" = FALSE,
                                                          "Yes" = TRUE
                                                           )
                                              ),
                                             actionButton("runSetup", "Run Setup Function")
                                             ),
                                           # Main panel for displaying outputs ----
                                           mainPanel(
                                             helpText("Use this intallation just for the first time. This will install packages and setup the conda environment!"),
                                             textOutput("python_loc_print"),
                                             textOutput("setupresult")
                                           )
                                  )
                       )
            )
  )
)

