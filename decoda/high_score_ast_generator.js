const fs = require('fs');
const path = require('path');
const esprima = require('esprima');
const mkdirp = require('mkdirp');
const crypto = require('crypto');
const archiver = require('archiver');

// u4feeu6539u53c2u6570u503cu4ee5u5904u7406high_score_files_by_typeu6587u4ef6u5939u4e2du7684u6587u4ef6uff0cu751fu6210ASTu5e76u538bu7f29
const inputRootPath = 'high_score_files_by_type';
const outputRootPath = 'high_score_files_ast';
const recursive = true;
const verbose = true; // u542fu7528u8be6u7ec6u8f93u51fau4ee5u67e5u770bu8fdbu5ea6
const maxFileSizeMB = 2; // u589eu52a0u6700u5927u6587u4ef6u5927u5c0fu9650u5236

// u6587u4ef6u8ba1u6570u5668
let processedCount = 0;
let errorCount = 0;
let duplicateCount = 0;
let skippedLargeFiles = 0;

// u5404u7c7bu578bu8ba1u6570
let typeStats = {
  bad: {
    train: 0,
    val: 0,
    test: 0
  },
  good: {
    train: 0,
    val: 0,
    test: 0
  }
};

// u7528u4e8eu8ddfu8e2au6587u4ef6u5185u5bb9u7684u54c8u5e0cuff0cu907fu514du91cdu590du5904u7406u76f8u540cu5185u5bb9
const contentHashes = new Set();

// u6e05u7a7au8f93u51fau76eeu5f55
if (fs.existsSync(outputRootPath)) {
  fs.rmSync(outputRootPath, { recursive: true, force: true });
}

// u521bu5efau8f93u51fau76eeu5f55u7ed3u6784
mkdirp.sync(outputRootPath);
const sourceTypes = ['bad', 'good'];
const dataTypes = ['train', 'val', 'test'];

// u521bu5efau76f8u540cu7684u76eeu5f55u7ed3u6784
for (const sourceType of sourceTypes) {
  for (const dataType of dataTypes) {
    const outputDir = path.join(outputRootPath, sourceType, dataType);
    mkdirp.sync(outputDir);
  }
}

/**
 * u5c06JavaScriptu4ee3u7801u89e3u6790u4e3aAST
 * @param {string} code JavaScriptu4ee3u7801
 * @param {Object} options Esprimau89e3u6790u9009u9879
 * @returns {Object} ASTu5bf9u8c61
 */
function parseToAST(code, options = {}) {
  try {
    // u4f7fu7528Esprimau89e3u6790u4ee3u7801
    const ast = esprima.parseScript(code, {
      comment: true,
      range: true,
      loc: true,
      ...options
    });
    return ast;
  } catch (error) {
    if (verbose) {
      console.error(`u89e3u6790u9519u8bef: ${error.message}`);
    }
    return null;
  }
}

/**
 * u68c0u67e5u6587u4ef6u5927u5c0f
 * @param {string} filePath u6587u4ef6u8defu5f84
 * @param {number} maxSizeMB u6700u5927u6587u4ef6u5927u5c0f(MB)
 * @returns {boolean} u662fu5426u5c0fu4e8eu6700u5927u5927u5c0f
 */
function checkFileSize(filePath, maxSizeMB = 1) {
  try {
    const stats = fs.statSync(filePath);
    const fileSizeMB = stats.size / (1024 * 1024);
    return fileSizeMB <= maxSizeMB;
  } catch (error) {
    if (verbose) {
      console.error(`u68c0u67e5u6587u4ef6u5927u5c0fu65f6u51fau9519: ${error.message}`);
    }
    return false;
  }
}

/**
 * u68c0u67e5u5185u5bb9u662fu5426u53efu80fdu662fJavaScriptu4ee3u7801
 * @param {string} content u6587u4ef6u5185u5bb9
 * @returns {boolean} u662fu5426u53efu80fdu662fJavaScriptu4ee3u7801
 */
function isLikelyJavaScript(content) {
  // u7b80u5355u68c0u67e5u662fu5426u5305u542bu5e38u89c1u7684JavaScriptu5173u952eu5b57u6216u8bedu6cd5
  const jsPatterns = [
    /function\s+\w+\s*\(/i,
    /var\s+\w+/i,
    /let\s+\w+/i,
    /const\s+\w+/i,
    /if\s*\(/i,
    /for\s*\(/i,
    /while\s*\(/i,
    /return\s+/i,
    /document\./i,
    /window\./i
  ];
  
  // u5982u679cu5339u914du4efbu4e00u6a21u5f0fuff0cu5219u53efu80fdu662fJavaScript
  return jsPatterns.some(pattern => pattern.test(content));
}

/**
 * u8ba1u7b97u5185u5bb9u7684MD5u54c8u5e0c
 * @param {string} content u8981u54c8u5e0cu7684u5185u5bb9
 * @returns {string} MD5u54c8u5e0cu503c
 */
function getContentHash(content) {
  return crypto.createHash('md5').update(content).digest('hex');
}

/**
 * u5904u7406u5355u4e2aJavaScriptu6587u4ef6
 * @param {string} filePath JavaScriptu6587u4ef6u8defu5f84
 * @param {string} outputDir u8f93u51fau76eeu5f55
 * @param {string} sourceType u6e90u7c7bu578b (badu6216good)
 * @param {string} dataType u6570u636eu7c7bu578b (train, valu6216test)
 */
function processFile(filePath, outputDir, sourceType, dataType) {
  // u68c0u67e5u6587u4ef6u5927u5c0fu662fu5426u8d85u8fc7u9650u5236
  if (!checkFileSize(filePath, maxFileSizeMB)) {
    skippedLargeFiles++;
    if (verbose) {
      console.warn(`u8df3u8fc7u5927u6587u4ef6: ${filePath} (u8d85u8fc7${maxFileSizeMB}MB)`);
    }
    return;
  }

  try {
    // u8bfbu53d6u6587u4ef6u5185u5bb9
    const code = fs.readFileSync(filePath, 'utf8');
    
    // u8ba1u7b97u5185u5bb9u54c8u5e0c
    const contentHash = getContentHash(code);
    
    // u5982u679cu5df2u7ecfu5904u7406u8fc7u76f8u540cu5185u5bb9u7684u6587u4ef6uff0cu5219u8df3u8fc7
    if (contentHashes.has(contentHash)) {
      duplicateCount++;
      if (verbose) {
        console.log(`u8df3u8fc7u91cdu590du6587u4ef6: ${filePath}`);
      }
      return;
    }
    
    // u68c0u67e5u662fu5426u53efu80fdu662fJavaScriptu4ee3u7801
    if (!isLikelyJavaScript(code)) {
      if (verbose) {
        console.warn(`u8df3u8fc7u975eJavaScriptu6587u4ef6: ${filePath}`);
      }
      return;
    }
    
    // u89e3u6790u4e3aAST
    const ast = parseToAST(code);
    
    if (ast) {
      // u6dfbu52a0u5185u5bb9u54c8u5e0cu5230u5df2u5904u7406u96c6u5408
      contentHashes.add(contentHash);
      
      // u751fu6210u6587u4ef6u540d
      const fileName = path.basename(filePath);
      
      // u751fu6210ASTu8f93u51fau6587u4ef6u8defu5f84
      const astOutputFilePath = path.join(outputDir, `${fileName}.ast.json`);
      
      // u5c06ASTu4fddu5b58u4e3aJSONu6587u4ef6
      fs.writeFileSync(astOutputFilePath, JSON.stringify(ast, null, 2), 'utf8');
      
      if (verbose) {
        console.log(`u5df2u751fu6210AST: ${astOutputFilePath}`);
      }
      
      processedCount++;
      typeStats[sourceType][dataType]++;
      
      // u6bcfu5904u740610u4e2au6587u4ef6u8f93u51fau4e00u6b21u8fdbu5ea6
      if (processedCount % 10 === 0) {
        console.log(`u5df2u5904u7406 ${processedCount} u4e2au6587u4ef6...`);
      }
    } else {
      errorCount++;
      if (verbose) {
        console.error(`u65e0u6cd5u89e3u6790u6587u4ef6: ${filePath}`);
      }
    }
  } catch (error) {
    if (verbose) {
      console.error(`u5904u7406u6587u4ef6 ${filePath} u65f6u51fau9519: ${error.message}`);
    }
    errorCount++;
  }
}

/**
 * u5904u7406high_score_files_by_typeu76eeu5f55u4e2du7684u6240u6709u6587u4ef6
 */
function processHighScoreFiles() {
  for (const sourceType of sourceTypes) {
    for (const dataType of dataTypes) {
      const inputDir = path.join(inputRootPath, sourceType, dataType);
      const outputDir = path.join(outputRootPath, sourceType, dataType);
      
      // u68c0u67e5u8f93u5165u76eeu5f55u662fu5426u5b58u5728
      if (!fs.existsSync(inputDir)) {
        console.log(`u8df3u8fc7u76eeu5f55uff08u4e0du5b58u5728uff09: ${inputDir}`);
        continue;
      }

      console.log(`u5904u7406u76eeu5f55: ${inputDir}`);
      
      try {
        const files = fs.readdirSync(inputDir);
        
        for (const file of files) {
          const filePath = path.join(inputDir, file);
          
          // u68c0u67e5u662fu5426u662fu6587u4ef6
          if (fs.statSync(filePath).isFile()) {
            processFile(filePath, outputDir, sourceType, dataType);
          }
        }
      } catch (error) {
        console.error(`u8bfbu53d6u76eeu5f55 ${inputDir} u65f6u51fau9519: ${error.message}`);
      }
    }
  }
}

/**
 * u538bu7f29u76eeu5f55u5230zipu6587u4ef6
 * @param {string} sourceDir u8981u538bu7f29u7684u76eeu5f55
 * @param {string} outputFilePath u8f93u51fau6587u4ef6u8defu5f84
 * @returns {Promise} u8fd4u56deu4e00u4e2aPromise
 */
function zipDirectory(sourceDir, outputFilePath) {
  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream(outputFilePath);
    const archive = archiver('zip', {
      zlib: { level: 9 } // u8bbeu7f6eu538bu7f29u7ea7u522b
    });
    
    output.on('close', () => {
      console.log(`u538bu7f29u5b8cu6210uff0cu603bu5927u5c0f: ${archive.pointer()} u5b57u8282`);
      resolve();
    });
    
    archive.on('error', (err) => {
      reject(err);
    });
    
    archive.pipe(output);
    archive.directory(sourceDir, false);
    archive.finalize();
  });
}

/**
 * u4e3Bu51fdu6570
 */
async function main() {
  console.log(`u5f00u59cbu5904u7406high_score_files_by_typeu6587u4ef6u5939u4e2du7684JavaScriptu6587u4ef6...`);
  console.log(`u8f93u5165u8defu5f84: ${inputRootPath}`);
  console.log(`ASTu8f93u51fau8defu5f84: ${outputRootPath}`);
  console.log(`u6700u5927u6587u4ef6u5927u5c0f: ${maxFileSizeMB} MB`);
  
  try {
    // u68c0u67e5u8f93u5165u8defu5f84u662fu5426u5b58u5728
    if (!fs.existsSync(inputRootPath)) {
      console.error(`u8f93u5165u8defu5f84 ${inputRootPath} u4e0du5b58u5728!`);
      process.exit(1);
    }
    
    console.log('u5f00u59cbu5904u7406uff0cu8bf7u7a0du5019...');
    
    // u5904u7406u6240u6709u6587u4ef6
    processHighScoreFiles();
    
    // u521bu5efau7edfu8ba1u6587u4ef6
    const statsData = {
      processed: processedCount,
      errors: errorCount,
      duplicates: duplicateCount,
      skippedLarge: skippedLargeFiles,
      typeStats: typeStats
    };
    
    // u4fddu5b58u7edfu8ba1u4fe1u606fu5230ASTu76eeu5f55
    fs.writeFileSync(
      path.join(outputRootPath, 'stats.json'), 
      JSON.stringify(statsData, null, 2), 
      'utf8'
    );
    
    // u538bu7f29u751fu6210u7684ASTu6587u4ef6
    console.log(`u5f00u59cbu538bu7f29ASTu6587u4ef6...`);
    await zipDirectory(outputRootPath, `${outputRootPath}.zip`);
    
    console.log(`u5904u7406u5b8cu6210!`);
    console.log(`u6210u529fu5904u7406: ${processedCount} u4e2au6587u4ef6`);
    console.log(`u8df3u8fc7u91cdu590d: ${duplicateCount} u4e2au6587u4ef6`);
    console.log(`u8df3u8fc7u5927u6587u4ef6: ${skippedLargeFiles} u4e2au6587u4ef6`);
    console.log(`u5904u7406u5931u8d25: ${errorCount} u4e2au6587u4ef6`);
    
    // u8f93u51fau5206u7c7bu7edfu8ba1
    console.log(`\nu5206u7c7bu7edfu8ba1:`);
    for (const sourceType of sourceTypes) {
      for (const dataType of dataTypes) {
        const count = typeStats[sourceType][dataType];
        console.log(`${sourceType}_${dataType}: ${count} u4e2au6587u4ef6`);
      }
    }
    
    // u538bu7f29u6587u4ef6u5730u5740
    console.log(`\nASTu6587u4ef6u5df2u538bu7f29u5230: ${outputRootPath}.zip`);
  } catch (error) {
    console.error(`u6267u884cu65f6u51fau9519: ${error.message}`);
    process.exit(1);
  }
}

// u6267u884cu4e3Bu51fdu6570
main(); 